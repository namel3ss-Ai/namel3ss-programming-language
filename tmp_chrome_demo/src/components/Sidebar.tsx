import { Link, useLocation } from "react-router-dom";
import { useState } from "react";

export interface NavItem {
  id: string;
  label: string;
  route?: string;
  icon?: string;
  badge?: { text: string; variant?: string };
  action?: string;
  condition?: string;
  children?: NavItem[];
}

export interface NavSection {
  id: string;
  label: string;
  items: string[];  // Nav item IDs
  collapsible: boolean;
  collapsed_by_default: boolean;
}

export interface SidebarProps {
  items: NavItem[];
  sections: NavSection[];
  collapsible?: boolean;
  collapsed_by_default?: boolean;
  width?: string;
  position?: "left" | "right";
  validated_routes?: string[];
}

export default function Sidebar({
  items,
  sections,
  collapsible = false,
  collapsed_by_default = false,
  width = "normal",
  position = "left",
}: SidebarProps) {
  const [collapsed, setCollapsed] = useState(collapsed_by_default);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());
  const location = useLocation();

  const toggleSection = (sectionId: string) => {
    setExpandedSections(prev => {
      const next = new Set(prev);
      if (next.has(sectionId)) {
        next.delete(sectionId);
      } else {
        next.add(sectionId);
      }
      return next;
    });
  };

  const renderNavItem = (item: NavItem, depth = 0) => {
    const isActive = item.route && location.pathname === item.route;
    const hasChildren = item.children && item.children.length > 0;
    const [expanded, setExpanded] = useState(false);

    return (
      <li key={item.id} style={{ marginLeft: `${depth * 1}rem` }}>
        <div className={`nav-item ${isActive ? 'active' : ''}`}>
          {item.route ? (
            <Link to={item.route} className="nav-link">
              {item.icon && <span className="nav-icon">{item.icon}</span>}
              {!collapsed && <span className="nav-label">{item.label}</span>}
              {item.badge && !collapsed && (
                <span className={`nav-badge ${item.badge.variant || ''}`}>
                  {item.badge.text}
                </span>
              )}
            </Link>
          ) : (
            <span className="nav-text">
              {item.icon && <span className="nav-icon">{item.icon}</span>}
              {!collapsed && <span className="nav-label">{item.label}</span>}
            </span>
          )}
          {hasChildren && !collapsed && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="nav-expand-btn"
              aria-label={expanded ? "Collapse" : "Expand"}
            >
              {expanded ? "▼" : "▶"}
            </button>
          )}
        </div>
        {hasChildren && expanded && !collapsed && (
          <ul className="nav-children">
            {item.children!.map(child => renderNavItem(child, depth + 1))}
          </ul>
        )}
      </li>
    );
  };

  const widthClass = width === "narrow" ? "sidebar-narrow" : width === "wide" ? "sidebar-wide" : "";
  const positionClass = position === "right" ? "sidebar-right" : "";
  const collapsedClass = collapsed ? "sidebar-collapsed" : "";

  return (
    <aside
      className={`sidebar ${widthClass} ${positionClass} ${collapsedClass}`}
      role="navigation"
      aria-label="Main navigation"
    >
      {collapsible && (
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="sidebar-toggle"
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? "☰" : "×"}
        </button>
      )}

      {sections.length > 0 ? (
        sections.map(section => {
          const sectionExpanded = expandedSections.has(section.id) || !section.collapsed_by_default;
          const sectionItems = items.filter(item => section.items.includes(item.id));

          return (
            <div key={section.id} className="nav-section">
              {section.collapsible ? (
                <button
                  onClick={() => toggleSection(section.id)}
                  className="nav-section-header"
                >
                  {!collapsed && <span>{section.label}</span>}
                  {!collapsed && <span>{sectionExpanded ? "▼" : "▶"}</span>}
                </button>
              ) : (
                <div className="nav-section-header">
                  {!collapsed && <span>{section.label}</span>}
                </div>
              )}
              {sectionExpanded && (
                <ul className="nav-list">
                  {sectionItems.map(item => renderNavItem(item))}
                </ul>
              )}
            </div>
          );
        })
      ) : (
        <ul className="nav-list">
          {items.map(item => renderNavItem(item))}
        </ul>
      )}
    </aside>
  );
}
