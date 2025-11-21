-- Database schema for {{ project_name }}
-- PostgreSQL 14+

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create {{ table_name }} table
CREATE TABLE IF NOT EXISTS {{ table_name }} (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    quantity INTEGER NOT NULL DEFAULT 0 CHECK (quantity >= 0),
    price NUMERIC(10, 2) NOT NULL DEFAULT 0.00 CHECK (price >= 0),
    is_active BOOLEAN NOT NULL DEFAULT true,
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ,
    
    -- Multi-tenancy support (optional)
    tenant_id VARCHAR(255)
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_{{ table_name }}_name ON {{ table_name }} (name);
CREATE INDEX IF NOT EXISTS idx_{{ table_name }}_is_active ON {{ table_name }} (is_active) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_{{ table_name }}_deleted_at ON {{ table_name }} (deleted_at);
CREATE INDEX IF NOT EXISTS idx_{{ table_name }}_created_at ON {{ table_name }} (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_{{ table_name }}_tenant_id ON {{ table_name }} (tenant_id) WHERE tenant_id IS NOT NULL;

-- GIN index for tags array searching
CREATE INDEX IF NOT EXISTS idx_{{ table_name }}_tags ON {{ table_name }} USING GIN (tags);

-- GIN index for JSONB metadata searching
CREATE INDEX IF NOT EXISTS idx_{{ table_name }}_metadata ON {{ table_name }} USING GIN (metadata);

-- Composite index for common filtered queries
CREATE INDEX IF NOT EXISTS idx_{{ table_name }}_active_created ON {{ table_name }} (is_active, created_at DESC) 
    WHERE deleted_at IS NULL;

-- Function to auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update updated_at on row update
DROP TRIGGER IF EXISTS update_{{ table_name }}_updated_at ON {{ table_name }};
CREATE TRIGGER update_{{ table_name }}_updated_at
    BEFORE UPDATE ON {{ table_name }}
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Optional: Add comment documentation
COMMENT ON TABLE {{ table_name }} IS 'Core {{ entity_name }} entities with soft delete support';
COMMENT ON COLUMN {{ table_name }}.id IS 'Unique identifier (UUID v4)';
COMMENT ON COLUMN {{ table_name }}.name IS '{{ entity_name }} name (required, indexed)';
COMMENT ON COLUMN {{ table_name }}.description IS 'Optional description';
COMMENT ON COLUMN {{ table_name }}.quantity IS 'Available quantity (non-negative)';
COMMENT ON COLUMN {{ table_name }}.price IS 'Unit price with 2 decimal precision (non-negative)';
COMMENT ON COLUMN {{ table_name }}.is_active IS 'Active status flag';
COMMENT ON COLUMN {{ table_name }}.tags IS 'Array of tags for categorization (indexed with GIN)';
COMMENT ON COLUMN {{ table_name }}.metadata IS 'JSON metadata for extensibility (indexed with GIN)';
COMMENT ON COLUMN {{ table_name }}.created_at IS 'Creation timestamp (auto-set)';
COMMENT ON COLUMN {{ table_name }}.updated_at IS 'Last update timestamp (auto-updated)';
COMMENT ON COLUMN {{ table_name }}.deleted_at IS 'Soft delete timestamp (NULL = not deleted)';
COMMENT ON COLUMN {{ table_name }}.tenant_id IS 'Tenant identifier for multi-tenancy (optional)';

-- Sample data for development (optional - comment out for production)
-- INSERT INTO {{ table_name }} (name, description, quantity, price, is_active, tags, metadata)
-- VALUES
--     ('Sample Widget', 'A sample widget for testing', 100, 29.99, true, 
--      ARRAY['electronics', 'hardware'], '{"color": "blue", "weight": "1kg"}'::JSONB),
--     ('Premium Service', 'Premium tier service offering', 0, 99.99, true,
--      ARRAY['service', 'premium'], '{"tier": "premium", "billing": "monthly"}'::JSONB),
--     ('Test Item', 'Item for testing purposes', 50, 9.99, false,
--      ARRAY['test'], '{}'::JSONB);
