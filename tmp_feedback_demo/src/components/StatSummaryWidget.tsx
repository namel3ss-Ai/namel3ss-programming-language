import React from 'react';
import { LineChart, Line, ResponsiveContainer } from 'recharts';

interface SparklineConfig {
  data: string;
  color?: string;
  height?: string;
}

interface StatSummaryWidgetConfig {
  id: string;
  type: 'stat_summary';
  title?: string;
  label: string;
  value: {
    field?: string;
    text?: string;
    format?: string;
  };
  delta?: {
    field: string;
    format?: string;
    showSign?: boolean;
  };
  trend?: {
    field: string;
    upIsGood?: boolean;
  };
  sparkline?: SparklineConfig;
  comparison?: string;
}

interface StatSummaryWidgetProps {
  widget: StatSummaryWidgetConfig;
  data: unknown;
}

function formatValue(value: any, format?: string): string {
  if (value === null || value === undefined) return '—';

  if (format === 'currency') {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(Number(value));
  }
  if (format === 'percentage') {
    return `${Number(value).toFixed(1)}%`;
  }
  if (format === 'number') {
    return new Intl.NumberFormat('en-US').format(Number(value));
  }
  if (format === 'compact') {
    const num = Number(value);
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return String(num);
  }

  return String(value);
}

function getTrendIcon(trend: 'up' | 'down' | 'neutral'): string {
  if (trend === 'up') return '↑';
  if (trend === 'down') return '↓';
  return '→';
}

function getTrendColor(trend: 'up' | 'down' | 'neutral', upIsGood: boolean): string {
  if (trend === 'neutral') return '#6b7280';
  if (trend === 'up') return upIsGood ? '#10b981' : '#ef4444';
  return upIsGood ? '#ef4444' : '#10b981';
}

export function StatSummaryWidget({ widget, data }: StatSummaryWidgetProps) {
  const item = data && typeof data === 'object' ? data : {};

  const value = widget.value.field
    ? (item as any)[widget.value.field]
    : widget.value.text || '—';

  const delta = widget.delta
    ? (item as any)[widget.delta.field]
    : null;

  const trendValue = widget.trend
    ? (item as any)[widget.trend.field]
    : null;

  let trend: 'up' | 'down' | 'neutral' = 'neutral';
  if (trendValue !== null && trendValue !== undefined) {
    if (Number(trendValue) > 0) trend = 'up';
    else if (Number(trendValue) < 0) trend = 'down';
  }

  const sparklineData = widget.sparkline
    ? ((item as any)[widget.sparkline.data] as any[] | undefined)
    : undefined;

  return (
    <div
      className="n3-stat-summary"
      style={{
        padding: '1.5rem',
        border: '1px solid #e5e7eb',
        borderRadius: '8px',
        backgroundColor: '#fff',
        minWidth: '200px',
      }}
    >
      {widget.title && (
        <h3 style={{ margin: '0 0 1rem', fontSize: '1.25rem', fontWeight: 600, color: '#111827' }}>
          {widget.title}
        </h3>
      )}

      <div className="stat-label" style={{ fontSize: '0.875rem', color: '#6b7280', marginBottom: '0.5rem' }}>
        {widget.label}
      </div>

      <div className="stat-value" style={{ fontSize: '2rem', fontWeight: 700, color: '#111827', marginBottom: '0.5rem' }}>
        {formatValue(value, widget.value.format)}
      </div>

      <div className="stat-details" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
        {widget.delta && delta !== null && delta !== undefined && (
          <div
            className="stat-delta"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.25rem',
              fontSize: '0.875rem',
              fontWeight: 600,
              color: getTrendColor(trend, widget.trend?.upIsGood !== false),
            }}
          >
            {getTrendIcon(trend)}
            {widget.delta.showSign !== false && delta > 0 ? '+' : ''}
            {formatValue(delta, widget.delta.format)}
          </div>
        )}

        {widget.comparison && (
          <div className="stat-comparison" style={{ fontSize: '0.75rem', color: '#9ca3af' }}>
            {widget.comparison}
          </div>
        )}
      </div>

      {sparklineData && Array.isArray(sparklineData) && sparklineData.length > 0 && (
        <div className="stat-sparkline" style={{ marginTop: '1rem', height: widget.sparkline?.height || '40px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={sparklineData}>
              <Line
                type="monotone"
                dataKey="value"
                stroke={widget.sparkline?.color || '#3b82f6'}
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
