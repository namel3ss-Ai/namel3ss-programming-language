"""
Enhanced React component generators with data binding support.

This module extends the base components.py with data-bound versions that use
the DatasetClient hooks (useDataset, useDatasetMutation) for:
- Dynamic tables with inline editing
- Charts with realtime updates
- Forms bound to datasets for create/update operations
"""

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from namel3ss.ir import ComponentSpec

from .utils import write_file


def write_bound_table_widget(components_dir: Path) -> None:
    """Generate BoundTableWidget.tsx for tables with dataset binding."""
    content = textwrap.dedent(
        """
        import { useState } from "react";
        import type { TableWidgetConfig } from "../lib/n3Client";
        import { useDataset, useDatasetMutation } from "../lib/datasetClient";

        interface BoundTableWidgetProps {
          widget: TableWidgetConfig & { binding?: any };
          datasetName: string;
          editable?: boolean;
          enableCreate?: boolean;
          enableUpdate?: boolean;
          enableDelete?: boolean;
        }

        export default function BoundTableWidget({
          widget,
          datasetName,
          editable = false,
          enableCreate = false,
          enableUpdate = false,
          enableDelete = false,
        }: BoundTableWidgetProps) {
          const [page, setPage] = useState(1);
          const [pageSize] = useState(widget.binding?.page_size || 50);
          const [sortBy, setSortBy] = useState<string | undefined>();
          const [sortOrder, setSortOrder] = useState<"asc" | "desc">("asc");
          const [search, setSearch] = useState("");
          const [editingRow, setEditingRow] = useState<number | null>(null);
          const [editData, setEditData] = useState<Record<string, any>>({});
          
          const { data, loading, error, refetch } = useDataset(datasetName, {
            page,
            page_size: pageSize,
            sort_by: sortBy,
            sort_order: sortOrder,
            search,
          });
          
          const { update, delete: deleteRecord } = useDatasetMutation(datasetName);
          
          const rows = data?.data || [];
          const columns = widget.columns && widget.columns.length
            ? widget.columns
            : rows.length ? Object.keys(rows[0]) : [];
          
          const handleSort = (column: string) => {
            if (sortBy === column) {
              setSortOrder(sortOrder === "asc" ? "desc" : "asc");
            } else {
              setSortBy(column);
              setSortOrder("asc");
            }
          };
          
          const handleEdit = (rowIndex: number, row: any) => {
            setEditingRow(rowIndex);
            setEditData({ ...row });
          };
          
          const handleSave = async (rowId: string | number) => {
            const result = await update(rowId, editData);
            if (result) {
              setEditingRow(null);
              setEditData({});
              refetch();
            }
          };
          
          const handleCancel = () => {
            setEditingRow(null);
            setEditData({});
          };
          
          const handleDelete = async (rowId: string | number) => {
            if (confirm("Are you sure you want to delete this record?")) {
              const success = await deleteRecord(rowId);
              if (success) {
                refetch();
              }
            }
          };
          
          if (loading && !data) {
            return (
              <section className="n3-widget">
                <h3>{widget.title}</h3>
                <div className="n3-loading">Loading...</div>
              </section>
            );
          }
          
          if (error) {
            return (
              <section className="n3-widget">
                <h3>{widget.title}</h3>
                <div className="n3-error">Error: {error.message}</div>
              </section>
            );
          }
          
          return (
            <section className="n3-widget">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                <h3 style={{ margin: 0 }}>{widget.title}</h3>
                <input
                  type="search"
                  placeholder="Search..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  style={{ padding: "0.5rem", borderRadius: "4px", border: "1px solid #ccc" }}
                />
              </div>
              
              {rows.length ? (
                <div style={{ overflowX: "auto" }}>
                  <table className="n3-table">
                    <thead>
                      <tr>
                        {columns.map((column) => (
                          <th
                            key={column}
                            onClick={() => handleSort(column)}
                            style={{ cursor: "pointer", userSelect: "none" }}
                          >
                            {column}
                            {sortBy === column && (sortOrder === "asc" ? " ↑" : " ↓")}
                          </th>
                        ))}
                        {(enableUpdate || enableDelete) && <th>Actions</th>}
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((row: any, idx: number) => {
                        const isEditing = editingRow === idx;
                        const rowId = row.id || row.rowId || idx;
                        
                        return (
                          <tr key={rowId}>
                            {columns.map((column) => (
                              <td key={column}>
                                {isEditing && enableUpdate ? (
                                  <input
                                    type="text"
                                    value={editData[column] ?? ""}
                                    onChange={(e) => setEditData({ ...editData, [column]: e.target.value })}
                                    style={{ width: "100%", padding: "0.25rem" }}
                                  />
                                ) : (
                                  String(row[column] ?? "")
                                )}
                              </td>
                            ))}
                            {(enableUpdate || enableDelete) && (
                              <td>
                                {isEditing ? (
                                  <>
                                    <button onClick={() => handleSave(rowId)} style={{ marginRight: "0.5rem" }}>
                                      Save
                                    </button>
                                    <button onClick={handleCancel}>Cancel</button>
                                  </>
                                ) : (
                                  <>
                                    {enableUpdate && (
                                      <button onClick={() => handleEdit(idx, row)} style={{ marginRight: "0.5rem" }}>
                                        Edit
                                      </button>
                                    )}
                                    {enableDelete && (
                                      <button onClick={() => handleDelete(rowId)} className="n3-btn-danger">
                                        Delete
                                      </button>
                                    )}
                                  </>
                                )}
                              </td>
                            )}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                  
                  {/* Pagination controls */}
                  {data && (
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "1rem" }}>
                      <div>
                        Showing {rows.length} of {data.total} records
                      </div>
                      <div>
                        <button
                          onClick={() => setPage(Math.max(1, page - 1))}
                          disabled={page === 1}
                          style={{ marginRight: "0.5rem" }}
                        >
                          Previous
                        </button>
                        <span style={{ marginRight: "0.5rem" }}>Page {page}</span>
                        <button
                          onClick={() => setPage(page + 1)}
                          disabled={!data.has_more}
                        >
                          Next
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div>No data available</div>
              )}
            </section>
          );
        }
        """
    ).strip() + "\n"
    write_file(components_dir / "BoundTableWidget.tsx", content)


def write_bound_chart_widget(components_dir: Path) -> None:
    """Generate BoundChartWidget.tsx for charts with dataset binding."""
    content = textwrap.dedent(
        """
        import type { ChartWidgetConfig } from "../lib/n3Client";
        import { useDataset } from "../lib/datasetClient";

        interface BoundChartWidgetProps {
          widget: ChartWidgetConfig & { binding?: any };
          datasetName: string;
        }

        export default function BoundChartWidget({ widget, datasetName }: BoundChartWidgetProps) {
          const { data, loading, error } = useDataset(datasetName, {
            page: 1,
            page_size: 1000, // Load more data for charts
          });
          
          if (loading && !data) {
            return (
              <section className="n3-widget">
                <h3>{widget.title}</h3>
                <div className="n3-loading">Loading chart data...</div>
              </section>
            );
          }
          
          if (error) {
            return (
              <section className="n3-widget">
                <h3>{widget.title}</h3>
                <div className="n3-error">Error: {error.message}</div>
              </section>
            );
          }
          
          const rows = data?.data || [];
          const xField = widget.x || "x";
          const yField = widget.y || "y";
          
          return (
            <section className="n3-widget">
              <h3>{widget.title}</h3>
              {rows.length ? (
                <div className="n3-chart">
                  <div style={{ marginBottom: "0.75rem" }}>
                    <strong>Chart Type: {widget.chartType || "bar"}</strong>
                  </div>
                  <ul style={{ listStyle: "none", paddingLeft: 0 }}>
                    {rows.map((row: any, idx: number) => (
                      <li key={idx} style={{ marginBottom: "0.5rem" }}>
                        <span style={{ fontWeight: 500 }}>{row[xField]}:</span>{" "}
                        {row[yField]}
                      </li>
                    ))}
                  </ul>
                  <div style={{ fontSize: "0.875rem", color: "#666", marginTop: "1rem" }}>
                    Total data points: {rows.length}
                    {data && data.total > rows.length && ` (showing first ${rows.length} of ${data.total})`}
                  </div>
                </div>
              ) : (
                <div>No data available for chart</div>
              )}
            </section>
          );
        }
        """
    ).strip() + "\n"
    write_file(components_dir / "BoundChartWidget.tsx", content)


def write_bound_form_widget(components_dir: Path) -> None:
    """Generate BoundFormWidget.tsx for forms with dataset binding."""
    content = textwrap.dedent(
        """
        import { FormEvent, useState, useEffect } from "react";
        import type { FormWidgetConfig } from "../lib/n3Client";
        import { useDatasetMutation } from "../lib/datasetClient";
        import { useToast } from "./Toast";

        interface BoundFormWidgetProps {
          widget: FormWidgetConfig & { binding?: any };
          datasetName: string;
          recordId?: string | number; // For update mode
          onSuccess?: () => void;
        }

        export default function BoundFormWidget({
          widget,
          datasetName,
          recordId,
          onSuccess,
        }: BoundFormWidgetProps) {
          const toast = useToast();
          const [formData, setFormData] = useState<Record<string, any>>({});
          const { create, update, loading, error } = useDatasetMutation(datasetName);
          
          const isUpdateMode = recordId !== undefined;
          
          const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
            event.preventDefault();
            
            try {
              let result;
              if (isUpdateMode) {
                result = await update(recordId, formData);
              } else {
                result = await create(formData);
              }
              
              if (result) {
                toast.show(widget.successMessage || (isUpdateMode ? "Record updated" : "Record created"));
                setFormData({});
                event.currentTarget.reset();
                onSuccess?.();
              } else {
                toast.show(error?.message || "Failed to save record");
              }
            } catch (err) {
              console.error("Form submission failed", err);
              toast.show("Unable to submit form right now");
            }
          };
          
          const handleChange = (fieldName: string, value: any) => {
            setFormData((prev) => ({ ...prev, [fieldName]: value }));
          };
          
          return (
            <section className="n3-widget">
              <h3>{widget.title}</h3>
              <form onSubmit={handleSubmit} className="n3-form">
                {widget.fields.map((field) => {
                  const fieldType = field.type || "text";
                  const fieldName = field.name;
                  
                  return (
                    <div key={fieldName} className="n3-form-field" style={{ marginBottom: "1rem" }}>
                      <label htmlFor={fieldName} style={{ display: "block", marginBottom: "0.25rem", fontWeight: 500 }}>
                        {fieldName}
                      </label>
                      {fieldType === "textarea" ? (
                        <textarea
                          id={fieldName}
                          name={fieldName}
                          value={formData[fieldName] || ""}
                          onChange={(e) => handleChange(fieldName, e.target.value)}
                          rows={4}
                          style={{ width: "100%", padding: "0.5rem" }}
                        />
                      ) : (
                        <input
                          id={fieldName}
                          name={fieldName}
                          type={fieldType}
                          value={formData[fieldName] || ""}
                          onChange={(e) => handleChange(fieldName, e.target.value)}
                          style={{ width: "100%", padding: "0.5rem" }}
                        />
                      )}
                    </div>
                  );
                })}
                
                <div style={{ marginTop: "1.5rem" }}>
                  <button
                    type="submit"
                    disabled={loading}
                    className="n3-btn-primary"
                    style={{ padding: "0.75rem 1.5rem" }}
                  >
                    {loading ? "Saving..." : (isUpdateMode ? "Update" : "Create")}
                  </button>
                </div>
                
                {error && (
                  <div className="n3-error" style={{ marginTop: "1rem" }}>
                    {error.message}
                  </div>
                )}
              </form>
            </section>
          );
        }
        """
    ).strip() + "\n"
    write_file(components_dir / "BoundFormWidget.tsx", content)


def generate_bound_components(components_dir: Path) -> None:
    """Generate all data-bound widget components."""
    write_bound_table_widget(components_dir)
    write_bound_chart_widget(components_dir)
    write_bound_form_widget(components_dir)
