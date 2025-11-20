from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from n3_server.config import get_settings
from n3_server.api import graphs, shares, tools, policies, import_export

settings = get_settings()

# OpenTelemetry setup
trace.set_tracer_provider(TracerProvider())
tracer_provider = trace.get_tracer_provider()
otlp_exporter = OTLPSpanExporter(endpoint=settings.otlp_endpoint)
span_processor = BatchSpanProcessor(otlp_exporter)
tracer_provider.add_span_processor(span_processor)

# FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

# Include routers
app.include_router(graphs.router, prefix=f"{settings.api_prefix}/graphs", tags=["graphs"])
app.include_router(shares.router, prefix=f"{settings.api_prefix}/projects", tags=["shares"])
app.include_router(tools.router, prefix=f"{settings.api_prefix}/tools", tags=["tools"])
app.include_router(policies.router, prefix=f"{settings.api_prefix}", tags=["policies"])
app.include_router(import_export.router, prefix=f"{settings.api_prefix}/n3", tags=["import-export"])


@app.get("/")
async def root():
    return {"message": "N3 Graph API", "version": settings.api_version}


@app.get("/health")
async def health():
    return {"status": "ok"}
