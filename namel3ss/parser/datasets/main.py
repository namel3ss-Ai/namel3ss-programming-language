"""Main dataset parser composition."""

from __future__ import annotations

from .utils import UtilityParserMixin
from .profile import ProfileParserMixin
from .schema import SchemaParserMixin
from .features import FeatureParserMixin
from .targets import TargetParserMixin
from .quality import QualityParserMixin
from .transforms import TransformParserMixin
from .connectors import ConnectorParserMixin
from .core import CoreDatasetParserMixin


class DatasetParserMixin(
    UtilityParserMixin,
    ProfileParserMixin,
    SchemaParserMixin,
    FeatureParserMixin,
    TargetParserMixin,
    QualityParserMixin,
    TransformParserMixin,
    ConnectorParserMixin,
    CoreDatasetParserMixin,
):
    """
    Complete dataset parser combining all parsing modules.
    
    This parser handles comprehensive dataset definitions including:
    - Source connections (table, file, SQL, REST, etc.)
    - Data transformations and operations
    - Schema specifications with types and constraints
    - Feature engineering for ML
    - Target definitions for ML models
    - Quality checks and validation rules
    - Profiles and metadata
    - Operational configs (caching, refresh, streaming)
    
    Syntax Example:
        dataset "sales_data" from table sales:
            filter by: revenue > 0
            add column profit = revenue - cost
            group by: region, product_category
            order by: date desc
            
            schema:
                region: string
                    description: "Sales region"
                    nullable: false
                revenue: float
                    constraints:
                        min: 0
            
            feature "revenue_feature":
                role: numeric
                source: revenue
                expression: log(revenue + 1)
            
            target "high_value":
                kind: classification
                expression: revenue > 10000
                positive_class: "high"
            
            quality "revenue_check":
                condition: revenue >= 0
                severity: error
                message: "Revenue cannot be negative"
            
            profile:
                row_count: 10000
                column_count: 15
                freshness: "1 hour"
            
            auto refresh every 5 minutes
            tags: sales, analytics, production
    
    Data Sources:
        - table: Existing database tables
        - file: CSV, JSON, Parquet files
        - dataset: Reference other N3 datasets
        - sql: SQL connectors with table/view/query access
        - rest: REST API endpoints with configurable requests
        - Custom connector types
    
    Operations:
        - Filter: filter by CONDITION
        - Compute: add column NAME = EXPRESSION
        - Group: group by COLUMNS
        - Order: order by COLUMNS
        - Aggregate: sum/count/avg/min/max: EXPRESSION
        - Join: join [type] source NAME on CONDITION
        - Window: add column NAME = FUNCTION over FRAME
    
    Configuration:
        - schema: Column definitions with types and constraints
        - transform: Custom transformation steps
        - feature: ML feature engineering specifications
        - target: ML prediction targets
        - quality: Data quality validation rules
        - profile: Dataset statistics and metadata
        - metadata: Custom metadata key-value pairs
        - lineage: Data lineage tracking
        - cache: Caching policies
        - pagination: Pagination configuration
        - stream: Streaming data policies
        - reactive: Real-time reactivity
        - auto refresh: Automatic refresh intervals
        - tags: Categorization tags
    
    Advanced Features:
        - Window functions with frames (ROWS/RANGE BETWEEN)
        - Multi-source joins with type specifications
        - Complex transformations with expressions
        - Quality checks with thresholds and alerts
        - Feature roles (numeric, categorical, text, etc.)
        - Target types (classification, regression, ranking)
        - Connector options with nested configuration
        - Refresh policies (polling, webhook, event-driven)
    """
    pass
