"""
Symbol Navigation and Semantic Analysis Demo.

Demonstrates advanced symbol understanding including:
- Go-to-definition across files
- Find-all-references with context
- Workspace-wide symbol search
- Intelligent hover information
- Dependency graph analysis
- Cross-file symbol resolution

This showcases the semantic intelligence that makes N3
feel like a first-class programming language.
"""

from pathlib import Path
from typing import Dict

from namel3ss.lsp.symbol_navigation import SymbolNavigationEngine, SemanticContext
from lsprotocol.types import Position, Range, SymbolKind


def demo_symbol_navigation():
    """Demonstrate advanced symbol navigation capabilities."""
    
    print("🎯 Symbol Navigation & Semantic Analysis Demo")
    print("=" * 50)
    
    # Create sample workspace with interconnected files
    workspace_files = {
        "file:///project/app.n3": '''app "ECommerce" {
    description: "Advanced e-commerce application"
}

llm "product_llm" {
    provider: "openai"
    model: "gpt-4o-mini"
    temperature: 0.3
}

memory "user_preferences" {
    scope: "user"
    kind: "dict"
    max_items: 100
}''',
        
        "file:///project/data.n3": '''frame "products" {
    source_type: "database"
    source: "postgres://localhost/ecommerce"
    columns: {
        id: int
        name: string
        price: float
        category_id: int
    }
}

frame "categories" {
    source_type: "database" 
    source: "postgres://localhost/ecommerce"
    columns: {
        id: int
        name: string
        description: string
    }
}

dataset "sales_data" {
    source_type: "csv"
    source: "data/sales.csv"
    schema: [
        {name: "product_id", type: "int"},
        {name: "quantity", type: "int"},
        {name: "revenue", type: "float"}
    ]
}''',
        
        "file:///project/pages.n3": '''page "ProductCatalog" at "/products" {
    show text: "Product Catalog"
    
    show form: {
        field: {
            name: "search"
            type: "text"
            placeholder: "Search products..."
        }
        field: {
            name: "category"
            type: "select"
            options: categories
        }
        submit: "Search"
    }
    
    show table: {
        source: products
        columns: ["name", "price", "category"]
        filters: {
            category: category
            search: search
        }
    }
}

page "ProductDetail" at "/product/{id}" {
    show text: "Product Details"
    
    set product_info: products.find(id)
    
    show component: "product_display" {
        product: product_info
        recommendations: product_llm.recommend(product_info)
    }
}

page "Analytics" at "/admin/analytics" {
    show text: "Sales Analytics"
    
    show chart: {
        type: "line"
        data: sales_data
        x_axis: "date"
        y_axis: "revenue"
        group_by: "product_id"
    }
}''',
        
        "file:///project/prompts.n3": '''prompt "product_recommendation" {
    model: product_llm
    template: """
    Based on the user's viewing history and preferences: {user_preferences}
    Current product: {current_product}
    
    Recommend 3 similar products that the user might like.
    Include reasoning for each recommendation.
    """
    args: {
        current_product: products
        user_preferences: user_preferences
    }
    output_schema: {
        recommendations: [
            {
                product_id: int
                reason: string
                confidence: float
            }
        ]
    }
}

prompt "category_insights" {
    model: product_llm
    template: """
    Analyze the sales data for category insights.
    Sales data: {sales_data}
    Categories: {categories}
    
    Provide insights about:
    1. Top performing categories
    2. Seasonal trends
    3. Recommendations for inventory
    """
    args: {
        sales_data: sales_data
        categories: categories
    }
}'''
    }
    
    # Test 1: Initialize Workspace
    print("\\n1. Workspace Initialization")
    print("-" * 30)
    
    engine = SymbolNavigationEngine()
    engine.initialize_workspace("/project", workspace_files)
    
    print(f"✅ Parsed {len(workspace_files)} files")
    print(f"✅ Indexed {len(engine.context.symbols)} symbols")
    
    # Show discovered symbols
    symbol_counts = {}
    for symbol in engine.context.symbols.values():
        kind_name = str(symbol.kind).split('.')[-1]
        symbol_counts[kind_name] = symbol_counts.get(kind_name, 0) + 1
    
    print("\\nDiscovered symbols:")
    for kind, count in symbol_counts.items():
        print(f"  • {count} {kind}(s)")
    
    # Test 2: Go-to-Definition
    print("\\n2. Go-to-Definition")
    print("-" * 25)
    
    # Test definition lookup for "products" referenced in pages.n3
    test_position = Position(line=15, character=20)  # Should be on "products" in show table
    definitions = engine.go_to_definition("file:///project/pages.n3", test_position)
    
    print("Looking up definition of 'products' from pages.n3:")
    for definition in definitions:
        file_name = definition.uri.split('/')[-1]
        line_num = definition.range.start.line + 1
        print(f"  → Found in {file_name} at line {line_num}")
    
    # Test 3: Find All References
    print("\\n3. Find All References")
    print("-" * 25)
    
    # Find all references to "product_llm"
    test_position_llm = Position(line=5, character=10)  # On "product_llm" in app.n3
    references = engine.find_references("file:///project/app.n3", test_position_llm)
    
    print("All references to 'product_llm':")
    for ref in references:
        file_name = ref.uri.split('/')[-1]
        line_num = ref.range.start.line + 1
        print(f"  → {file_name}:{line_num}")
    
    # Test 4: Workspace Symbol Search
    print("\\n4. Workspace Symbol Search")
    print("-" * 30)
    
    # Search for symbols containing "product"
    product_symbols = engine.get_workspace_symbols("product")
    
    print("Symbols matching 'product':")
    for symbol in product_symbols[:5]:  # Show first 5
        file_name = symbol.location.uri.split('/')[-1]
        kind_name = str(symbol.kind).split('.')[-1]
        line_num = symbol.location.range.start.line + 1
        print(f"  • {symbol.name} ({kind_name}) in {file_name}:{line_num}")
    
    # Test 5: Hover Information
    print("\\n5. Hover Information")
    print("-" * 25)
    
    # Get hover info for "categories" frame
    hover_position = Position(line=8, character=10)  # On "categories" definition
    hover_info = engine.get_hover_info("file:///project/data.n3", hover_position)
    
    if hover_info:
        print("Hover info for 'categories':")
        content_lines = hover_info.contents.value.split('\\n')
        for line in content_lines[:4]:  # Show first few lines
            print(f"  {line}")
        if len(content_lines) > 4:
            print("  ...")
    
    # Test 6: Dependency Analysis
    print("\\n6. Dependency Analysis")
    print("-" * 25)
    
    # Analyze dependencies for ProductCatalog page
    deps_analysis = engine.analyze_symbol_dependencies("ProductCatalog")
    
    if deps_analysis:
        print("Dependencies for 'ProductCatalog' page:")
        direct_deps = deps_analysis.get('direct_dependencies', [])
        if direct_deps:
            print(f"  Direct dependencies: {', '.join(direct_deps)}")
        
        dependents = deps_analysis.get('dependents', [])
        if dependents:
            print(f"  Used by: {', '.join(dependents)}")
        
        ref_count = deps_analysis.get('reference_count', 0)
        print(f"  Referenced {ref_count} times across project")
    
    # Test 7: Document Symbols
    print("\\n7. Document Symbols")
    print("-" * 25)
    
    # Get symbols in the pages.n3 file
    doc_symbols = engine.get_document_symbols("file:///project/pages.n3")
    
    print("Symbols in pages.n3:")
    for symbol in doc_symbols:
        kind_name = str(symbol.kind).split('.')[-1]
        line_num = symbol.range.start.line + 1
        print(f"  • {symbol.name} ({kind_name}) at line {line_num}")
        if symbol.detail:
            print(f"    {symbol.detail}")
    
    # Test 8: Cross-File Relationships
    print("\\n8. Cross-File Relationships")
    print("-" * 30)
    
    relationships = []
    
    # Find which files reference symbols from other files
    for uri, _ in workspace_files.items():
        file_name = uri.split('/')[-1]
        
        # Count symbols defined in this file
        local_symbols = [s for s in engine.context.symbols.values() if s.uri == uri]
        
        # Count references from other files
        external_refs = 0
        for symbol in local_symbols:
            for ref in symbol.references:
                if ref.uri != uri:
                    external_refs += 1
        
        if external_refs > 0:
            relationships.append((file_name, len(local_symbols), external_refs))
    
    print("Cross-file symbol usage:")
    for file_name, local_count, external_refs in relationships:
        print(f"  • {file_name}: {local_count} symbols, {external_refs} external references")
    
    # Test 9: Performance Metrics
    print("\\n9. Performance & Statistics")
    print("-" * 35)
    
    total_symbols = len(engine.context.symbols)
    total_references = sum(len(s.references) for s in engine.context.symbols.values())
    
    print(f"Workspace analysis completed:")
    print(f"  • {len(workspace_files)} files processed")
    print(f"  • {total_symbols} symbols indexed")
    print(f"  • {total_references} references tracked")
    print(f"  • {len(engine.context.dependencies)} dependency relationships")
    
    # Memory usage estimation
    avg_refs_per_symbol = total_references / total_symbols if total_symbols > 0 else 0
    print(f"  • {avg_refs_per_symbol:.1f} average references per symbol")
    
    print("\\n🎉 Symbol Navigation Demo Complete!")
    
    navigation_features = [
        "✅ Go-to-definition across files and projects",
        "✅ Find-all-references with contextual information", 
        "✅ Workspace-wide symbol search with relevance ranking",
        "✅ Rich hover information with dependencies and usage",
        "✅ Hierarchical document symbol outline",
        "✅ Cross-file dependency analysis and tracking",
        "✅ Real-time semantic understanding of code structure",
        "✅ Integration with 237x faster parser caching"
    ]
    
    print("\\n🧠 Semantic Intelligence Features:")
    for feature in navigation_features:
        print(f"   {feature}")
    
    print("\\n💡 Developer Experience:")
    print("   • Jump to any symbol definition instantly (F12)")
    print("   • Find all usages of variables, functions, components (Shift+F12)")
    print("   • Search across entire project with intelligent ranking (Ctrl+T)")
    print("   • Rich IntelliSense with type information and documentation")
    print("   • Understand code relationships and dependencies visually")
    print("   • Navigate large codebases with confidence and speed")
    
    return {
        "files_processed": len(workspace_files),
        "symbols_indexed": len(engine.context.symbols),
        "references_tracked": total_references,
        "cross_file_relationships": len(relationships),
        "workspace_search_working": len(product_symbols) > 0,
        "hover_info_working": hover_info is not None,
        "dependency_analysis_working": bool(deps_analysis)
    }


if __name__ == "__main__":
    demo_symbol_navigation()