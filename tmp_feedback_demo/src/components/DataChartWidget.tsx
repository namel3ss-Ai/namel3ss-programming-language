import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  BarChart,
  AreaChart,
  PieChart,
  ScatterChart,
  Line,
  Bar,
  Area,
  Pie,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Cell,
} from 'recharts';

interface ChartSeries {
  dataKey: string;
  label: string;
  color?: string;
  type?: 'line' | 'bar' | 'area';
}

interface ChartConfig {
  type: 'line' | 'bar' | 'pie' | 'area' | 'scatter';
  xAxis?: string;
  yAxis?: string;
  series: ChartSeries[];
  legend?: boolean;
  grid?: boolean;
  height?: string;
}

interface EmptyStateConfig {
  icon?: string;
  title: string;
  message?: string;
  actionLabel?: string;
  actionLink?: string;
}

interface DataChartWidgetConfig {
  id: string;
  type: 'data_chart';
  title?: string;
  source: {
    kind: string;
    name: string;
  };
  chartConfig: ChartConfig;
  emptyState?: EmptyStateConfig;
}

interface DataChartWidgetProps {
  widget: DataChartWidgetConfig;
  data: unknown;
}

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];

function EmptyState({ config }: { config: EmptyStateConfig }) {
  return (
    <div className="n3-empty-state" style={{ textAlign: 'center', padding: '3rem 1rem' }}>
      {config.icon && (
        <div className="empty-state-icon" style={{ fontSize: '2rem', marginBottom: '1rem', opacity: 0.5 }}>
          📈
        </div>
      )}
      <h3 style={{ margin: '0 0 0.5rem', fontSize: '1.25rem', fontWeight: 600 }}>{config.title}</h3>
      {config.message && (
        <p style={{ margin: '0 0 1rem', color: '#666' }}>{config.message}</p>
      )}
      {config.actionLabel && config.actionLink && (
        <a href={config.actionLink} className="btn btn-primary">
          {config.actionLabel}
        </a>
      )}
    </div>
  );
}

export function DataChartWidget({ widget, data }: DataChartWidgetProps) {
  const chartData = Array.isArray(data) ? data : [];
  const { chartConfig } = widget;
  const height = parseInt(chartConfig.height || '300');

  if (chartData.length === 0 && widget.emptyState) {
    return <EmptyState config={widget.emptyState} />;
  }

  const renderChart = () => {
    const commonProps = {
      data: chartData,
      margin: { top: 5, right: 30, left: 20, bottom: 5 },
    };

    switch (chartConfig.type) {
      case 'line':
        return (
          <LineChart {...commonProps}>
            {chartConfig.grid && <CartesianGrid strokeDasharray="3 3" />}
            {chartConfig.xAxis && <XAxis dataKey={chartConfig.xAxis} />}
            {chartConfig.yAxis && <YAxis />}
            <Tooltip />
            {chartConfig.legend !== false && <Legend />}
            {chartConfig.series.map((series, idx) => (
              <Line
                key={series.dataKey}
                type="monotone"
                dataKey={series.dataKey}
                name={series.label}
                stroke={series.color || COLORS[idx % COLORS.length]}
                strokeWidth={2}
                dot={{ r: 3 }}
                activeDot={{ r: 5 }}
              />
            ))}
          </LineChart>
        );

      case 'bar':
        return (
          <BarChart {...commonProps}>
            {chartConfig.grid && <CartesianGrid strokeDasharray="3 3" />}
            {chartConfig.xAxis && <XAxis dataKey={chartConfig.xAxis} />}
            {chartConfig.yAxis && <YAxis />}
            <Tooltip />
            {chartConfig.legend !== false && <Legend />}
            {chartConfig.series.map((series, idx) => (
              <Bar
                key={series.dataKey}
                dataKey={series.dataKey}
                name={series.label}
                fill={series.color || COLORS[idx % COLORS.length]}
              />
            ))}
          </BarChart>
        );

      case 'area':
        return (
          <AreaChart {...commonProps}>
            {chartConfig.grid && <CartesianGrid strokeDasharray="3 3" />}
            {chartConfig.xAxis && <XAxis dataKey={chartConfig.xAxis} />}
            {chartConfig.yAxis && <YAxis />}
            <Tooltip />
            {chartConfig.legend !== false && <Legend />}
            {chartConfig.series.map((series, idx) => (
              <Area
                key={series.dataKey}
                type="monotone"
                dataKey={series.dataKey}
                name={series.label}
                stroke={series.color || COLORS[idx % COLORS.length]}
                fill={series.color || COLORS[idx % COLORS.length]}
                fillOpacity={0.6}
              />
            ))}
          </AreaChart>
        );

      case 'pie':
        return (
          <PieChart>
            <Pie
              data={chartData}
              dataKey={chartConfig.series[0]?.dataKey || 'value'}
              nameKey={chartConfig.xAxis || 'name'}
              cx="50%"
              cy="50%"
              outerRadius={Math.min(height * 0.4, 120)}
              label
            >
              {chartData.map((_, idx) => (
                <Cell key={`cell-${idx}`} fill={COLORS[idx % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            {chartConfig.legend !== false && <Legend />}
          </PieChart>
        );

      case 'scatter':
        return (
          <ScatterChart {...commonProps}>
            {chartConfig.grid && <CartesianGrid strokeDasharray="3 3" />}
            {chartConfig.xAxis && <XAxis dataKey={chartConfig.xAxis} name={chartConfig.xAxis} />}
            {chartConfig.yAxis && <YAxis dataKey={chartConfig.series[0]?.dataKey} name={chartConfig.series[0]?.label} />}
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            {chartConfig.legend !== false && <Legend />}
            {chartConfig.series.map((series, idx) => (
              <Scatter
                key={series.dataKey}
                name={series.label}
                dataKey={series.dataKey}
                fill={series.color || COLORS[idx % COLORS.length]}
              />
            ))}
          </ScatterChart>
        );

      default:
        return <div>Unsupported chart type: {chartConfig.type}</div>;
    }
  };

  return (
    <div className="n3-data-chart" style={{ width: '100%' }}>
      {widget.title && (
        <h2 style={{ margin: '0 0 1rem', fontSize: '1.5rem', fontWeight: 600 }}>{widget.title}</h2>
      )}

      <ResponsiveContainer width="100%" height={height}>
        {renderChart()}
      </ResponsiveContainer>
    </div>
  );
}
