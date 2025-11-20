"""
API endpoint for importing N3 source code into graph editor.

Parses N3 files and converts AST to graph JSON format.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from n3_server.db.session import get_db
from n3_server.db.models import Project
from n3_server.converter import N3ASTConverter
from nanoid import generate

router = APIRouter()


class ImportN3Request(BaseModel):
    """Request to import N3 source code."""
    source: str
    projectName: Optional[str] = None


class ImportN3Response(BaseModel):
    """Response from N3 import."""
    projectId: str
    name: str
    nodesCreated: int
    edgesCreated: int
    chains: int
    agents: int


@router.post("/import", response_model=ImportN3Response)
async def import_n3_source(
    request: ImportN3Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Import N3 source code and convert to graph JSON.
    
    Parses the N3 source, extracts AST nodes, and creates a new project
    with the graph representation.
    """
    try:
        # Parse N3 source (requires N3 parser integration)
        # TODO: Import and use namel3ss.parser to parse source
        # For now, we'll create a demo structure
        
        converter = N3ASTConverter()
        project_id = generate(size=12)
        project_name = request.projectName or "Imported Project"
        
        # TODO: Parse actual N3 source to extract chains, agents, prompts, etc.
        # parsed_ast = parse_n3_source(request.source)
        # chains = extract_chains(parsed_ast)
        # agents = extract_agents(parsed_ast)
        # prompts = extract_prompts(parsed_ast)
        # rags = extract_rags(parsed_ast)
        
        # For demo, create empty graph
        graph_json = converter.ast_to_graph_json(
            project_id=project_id,
            name=project_name,
            chains=[],
            agents=[],
            agent_graphs=[],
            prompts=[],
            rags=[],
        )
        
        # Create project in database
        project = Project(
            id=project_id,
            name=project_name,
            graph_data={
                "chains": graph_json.chains,
                "agents": graph_json.agents,
                "activeRootId": graph_json.activeRootId,
                "nodes": graph_json.nodes,
                "edges": graph_json.edges,
            },
            metadata={"imported_from_source": True}
        )
        
        db.add(project)
        await db.commit()
        await db.refresh(project)
        
        return ImportN3Response(
            projectId=project_id,
            name=project_name,
            nodesCreated=len(graph_json.nodes),
            edgesCreated=len(graph_json.edges),
            chains=len(graph_json.chains),
            agents=len(graph_json.agents),
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Import failed: {str(e)}")


@router.post("/export/{project_id}")
async def export_to_n3(
    project_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Export graph JSON back to N3 source code.
    
    Converts graph representation to N3 AST and generates source code.
    """
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    try:
        converter = N3ASTConverter()
        graph_data = project.graph_data or {}
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        
        # Convert graph to AST
        # TODO: Generate actual N3 source code from AST
        chain = converter.graph_json_to_chain(nodes, edges, project.name)
        
        # Generate N3 source (pseudo-code for now)
        n3_source = f"""# Generated from {project.name}

define chain {chain.name} {{
    input_key: "{chain.input_key}"
    
"""
        
        for idx, step in enumerate(chain.steps):
            n3_source += f"""    step step_{idx + 1} {{
        kind: {step.kind}
        target: "{step.target}"
        options: {step.options}
    }}
    
"""
        
        n3_source += "}\n"
        
        return {
            "projectId": project_id,
            "name": project.name,
            "source": n3_source,
            "stepsExported": len(chain.steps),
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Export failed: {str(e)}")
