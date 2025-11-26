"""
React chrome component generators (navigation & app chrome).

This module generates TypeScript React components for:
- Sidebar navigation with hierarchical items
- Navbar/topbar with branding and actions
- Breadcrumbs with auto-derivation
- Command palette with keyboard shortcuts
"""

import textwrap
from pathlib import Path
from typing import Dict, Any, List

from .utils import write_file


def write_sidebar_component(components_dir: Path) -> None:
    """Generate Sidebar.tsx with hierarchical navigation."""
    content = textwrap.dedent(
        """
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
        """
    ).strip() + "\n"
    write_file(components_dir / "Sidebar.tsx", content)


def write_navbar_component(components_dir: Path) -> None:
    """Generate Navbar.tsx with branding and actions."""
    content = textwrap.dedent(
        """
        import { useState, useRef, useEffect } from "react";
        import type { NavItem } from "./Sidebar";

        export interface NavbarAction {
          id: string;
          label?: string;
          icon?: string;
          type: "button" | "menu" | "toggle";
          action?: string;
          menu_items?: NavItem[];
          condition?: string;
        }

        export interface NavbarProps {
          logo?: string;
          title?: string;
          actions: NavbarAction[];
          position?: "top" | "bottom";
          sticky?: boolean;
          validated_actions?: string[];
        }

        export default function Navbar({
          logo,
          title,
          actions,
          position = "top",
          sticky = true,
        }: NavbarProps) {
          const [openMenus, setOpenMenus] = useState<Set<string>>(new Set());
          const menuRefs = useRef<Map<string, HTMLDivElement>>(new Map());

          useEffect(() => {
            const handleClickOutside = (event: MouseEvent) => {
              const clickedOutside = Array.from(menuRefs.current.values()).every(
                ref => !ref.contains(event.target as Node)
              );
              if (clickedOutside) {
                setOpenMenus(new Set());
              }
            };

            document.addEventListener('mousedown', handleClickOutside);
            return () => document.removeEventListener('mousedown', handleClickOutside);
          }, []);

          const toggleMenu = (actionId: string) => {
            setOpenMenus(prev => {
              const next = new Set(prev);
              if (next.has(actionId)) {
                next.delete(actionId);
              } else {
                next.add(actionId);
              }
              return next;
            });
          };

          const handleAction = (actionId?: string) => {
            if (actionId) {
              // Dispatch action event
              window.dispatchEvent(new CustomEvent('namel3ss:action', { 
                detail: { actionId } 
              }));
            }
          };

          const renderAction = (action: NavbarAction) => {
            const isMenuOpen = openMenus.has(action.id);

            if (action.type === "menu") {
              return (
                <div
                  key={action.id}
                  className="navbar-action navbar-menu"
                  ref={ref => {
                    if (ref) menuRefs.current.set(action.id, ref);
                  }}
                >
                  <button
                    onClick={() => toggleMenu(action.id)}
                    className="navbar-btn"
                    aria-haspopup="true"
                    aria-expanded={isMenuOpen}
                  >
                    {action.icon && <span className="navbar-icon">{action.icon}</span>}
                    {action.label && <span>{action.label}</span>}
                  </button>
                  {isMenuOpen && action.menu_items && action.menu_items.length > 0 && (
                    <div className="navbar-menu-dropdown" role="menu">
                      {action.menu_items.map(item => (
                        <button
                          key={item.id}
                          onClick={() => {
                            if (item.action) handleAction(item.action);
                            if (item.route) window.location.href = item.route;
                            setOpenMenus(new Set());
                          }}
                          className="navbar-menu-item"
                          role="menuitem"
                        >
                          {item.icon && <span className="navbar-icon">{item.icon}</span>}
                          <span>{item.label}</span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              );
            }

            return (
              <button
                key={action.id}
                onClick={() => handleAction(action.action)}
                className={`navbar-btn navbar-${action.type}`}
              >
                {action.icon && <span className="navbar-icon">{action.icon}</span>}
                {action.label && <span>{action.label}</span>}
              </button>
            );
          };

          const positionClass = position === "bottom" ? "navbar-bottom" : "";
          const stickyClass = sticky ? "navbar-sticky" : "";

          return (
            <nav className={`navbar ${positionClass} ${stickyClass}`} role="navigation">
              <div className="navbar-start">
                {logo && <img src={logo} alt="Logo" className="navbar-logo" />}
                {title && <span className="navbar-title">{title}</span>}
              </div>
              <div className="navbar-end">
                {actions.map(action => renderAction(action))}
              </div>
            </nav>
          );
        }
        """
    ).strip() + "\n"
    write_file(components_dir / "Navbar.tsx", content)


def write_breadcrumbs_component(components_dir: Path) -> None:
    """Generate Breadcrumbs.tsx with auto-derivation support."""
    content = textwrap.dedent(
        """
        import { Link, useLocation } from "react-router-dom";
        import { useMemo } from "react";

        export interface BreadcrumbItem {
          label: string;
          route?: string;
        }

        export interface BreadcrumbsProps {
          items?: BreadcrumbItem[];
          auto_derive?: boolean;
          separator?: string;
          derived_from_route?: string;
        }

        export default function Breadcrumbs({
          items = [],
          auto_derive = false,
          separator = "/",
          derived_from_route,
        }: BreadcrumbsProps) {
          const location = useLocation();

          const breadcrumbItems = useMemo(() => {
            if (auto_derive) {
              // Auto-derive from current route
              const pathSegments = location.pathname.split('/').filter(Boolean);
              return pathSegments.map((segment, index) => {
                const route = '/' + pathSegments.slice(0, index + 1).join('/');
                const label = segment.charAt(0).toUpperCase() + segment.slice(1).replace(/-/g, ' ');
                return { label, route };
              });
            }
            return items;
          }, [auto_derive, location.pathname, items]);

          if (breadcrumbItems.length === 0) {
            return null;
          }

          return (
            <nav className="breadcrumbs" aria-label="Breadcrumb">
              <ol className="breadcrumbs-list">
                {breadcrumbItems.map((item, index) => {
                  const isLast = index === breadcrumbItems.length - 1;
                  
                  return (
                    <li key={index} className="breadcrumbs-item">
                      {item.route && !isLast ? (
                        <Link to={item.route} className="breadcrumbs-link">
                          {item.label}
                        </Link>
                      ) : (
                        <span className="breadcrumbs-current" aria-current={isLast ? "page" : undefined}>
                          {item.label}
                        </span>
                      )}
                      {!isLast && (
                        <span className="breadcrumbs-separator" aria-hidden="true">
                          {separator}
                        </span>
                      )}
                    </li>
                  );
                })}
              </ol>
            </nav>
          );
        }
        """
    ).strip() + "\n"
    write_file(components_dir / "Breadcrumbs.tsx", content)


def write_command_palette_component(components_dir: Path) -> None:
    """Generate CommandPalette.tsx with keyboard navigation."""
    content = textwrap.dedent(
        """
        import { useState, useEffect, useCallback, useRef } from "react";
        import { useNavigate } from "react-router-dom";

        export interface CommandSource {
          type: "routes" | "actions" | "custom";
          filter?: string;
          custom_items?: Array<{ label: string; action?: string }>;
        }

        export interface CommandPaletteProps {
          shortcut?: string;
          sources: CommandSource[];
          placeholder?: string;
          max_results?: number;
          available_routes?: Array<{ label: string; path: string }>;
          available_actions?: Array<{ label: string; id: string }>;
        }

        interface Command {
          id: string;
          label: string;
          type: "route" | "action";
          target?: string;  // route path or action ID
        }

        export default function CommandPalette({
          shortcut = "ctrl+k",
          sources,
          placeholder = "Search commands...",
          max_results = 10,
          available_routes = [],
          available_actions = [],
        }: CommandPaletteProps) {
          const [isOpen, setIsOpen] = useState(false);
          const [query, setQuery] = useState("");
          const [selectedIndex, setSelectedIndex] = useState(0);
          const inputRef = useRef<HTMLInputElement>(null);
          const navigate = useNavigate();

          // Build command list from sources
          const allCommands = useMemo((): Command[] => {
            const commands: Command[] = [];

            sources.forEach(source => {
              if (source.type === "routes") {
                available_routes.forEach(route => {
                  commands.push({
                    id: `route:${route.path}`,
                    label: route.label,
                    type: "route",
                    target: route.path,
                  });
                });
              } else if (source.type === "actions") {
                available_actions.forEach(action => {
                  commands.push({
                    id: `action:${action.id}`,
                    label: action.label,
                    type: "action",
                    target: action.id,
                  });
                });
              } else if (source.type === "custom" && source.custom_items) {
                source.custom_items.forEach(item => {
                  commands.push({
                    id: `custom:${item.label}`,
                    label: item.label,
                    type: "action",
                    target: item.action,
                  });
                });
              }
            });

            return commands;
          }, [sources, available_routes, available_actions]);

          // Filter commands based on query
          const filteredCommands = useMemo(() => {
            if (!query) return allCommands.slice(0, max_results);

            const lowerQuery = query.toLowerCase();
            return allCommands
              .filter(cmd => cmd.label.toLowerCase().includes(lowerQuery))
              .slice(0, max_results);
          }, [query, allCommands, max_results]);

          // Handle keyboard shortcut
          useEffect(() => {
            const handleKeyDown = (e: KeyboardEvent) => {
              // Parse shortcut (e.g., "ctrl+k", "cmd+k")
              const parts = shortcut.toLowerCase().split('+');
              const hasCtrl = parts.includes('ctrl') || parts.includes('cmd');
              const key = parts[parts.length - 1];

              if (hasCtrl && (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === key) {
                e.preventDefault();
                setIsOpen(prev => !prev);
              }

              // Close on Escape
              if (e.key === 'Escape' && isOpen) {
                setIsOpen(false);
                setQuery("");
              }
            };

            window.addEventListener('keydown', handleKeyDown);
            return () => window.removeEventListener('keydown', handleKeyDown);
          }, [shortcut, isOpen]);

          // Focus input when opened
          useEffect(() => {
            if (isOpen && inputRef.current) {
              inputRef.current.focus();
            }
          }, [isOpen]);

          // Handle command navigation with arrow keys
          const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
            if (e.key === 'ArrowDown') {
              e.preventDefault();
              setSelectedIndex(prev => 
                prev < filteredCommands.length - 1 ? prev + 1 : prev
              );
            } else if (e.key === 'ArrowUp') {
              e.preventDefault();
              setSelectedIndex(prev => prev > 0 ? prev - 1 : prev);
            } else if (e.key === 'Enter') {
              e.preventDefault();
              const command = filteredCommands[selectedIndex];
              if (command) executeCommand(command);
            }
          }, [filteredCommands, selectedIndex]);

          const executeCommand = (command: Command) => {
            if (command.type === "route" && command.target) {
              navigate(command.target);
            } else if (command.type === "action" && command.target) {
              window.dispatchEvent(new CustomEvent('namel3ss:action', {
                detail: { actionId: command.target }
              }));
            }
            setIsOpen(false);
            setQuery("");
            setSelectedIndex(0);
          };

          if (!isOpen) return null;

          return (
            <>
              <div className="command-palette-overlay" onClick={() => setIsOpen(false)} />
              <div className="command-palette" role="dialog" aria-label="Command palette">
                <div className="command-palette-input-wrapper">
                  <input
                    ref={inputRef}
                    type="text"
                    className="command-palette-input"
                    placeholder={placeholder}
                    value={query}
                    onChange={e => {
                      setQuery(e.target.value);
                      setSelectedIndex(0);
                    }}
                    onKeyDown={handleKeyDown}
                    aria-label="Search commands"
                    aria-autocomplete="list"
                    aria-controls="command-list"
                  />
                </div>
                <ul id="command-list" className="command-palette-results" role="listbox">
                  {filteredCommands.length === 0 ? (
                    <li className="command-palette-no-results">No commands found</li>
                  ) : (
                    filteredCommands.map((command, index) => (
                      <li
                        key={command.id}
                        className={`command-palette-item ${index === selectedIndex ? 'selected' : ''}`}
                        onClick={() => executeCommand(command)}
                        role="option"
                        aria-selected={index === selectedIndex}
                      >
                        <span className="command-label">{command.label}</span>
                        <span className="command-type">{command.type}</span>
                      </li>
                    ))
                  )}
                </ul>
              </div>
            </>
          );
        }
        """
    ).strip() + "\n"
    write_file(components_dir / "CommandPalette.tsx", content)


def write_modal_component(components_dir: Path) -> None:
    """Generate Modal.tsx dialog component."""
    content = textwrap.dedent(
        """
        import { useState, useEffect, ReactNode } from "react";
        import { X } from "lucide-react";

        export interface ModalAction {
          label: string;
          action?: string;
          variant?: "default" | "primary" | "destructive" | "ghost" | "link";
          close?: boolean;
        }

        export interface ModalProps {
          id: string;
          title: string;
          description?: string;
          content: ReactNode[];
          actions: ModalAction[];
          size?: "sm" | "md" | "lg" | "xl" | "full";
          dismissible?: boolean;
          trigger?: string;
          isOpen?: boolean;
          onClose?: () => void;
        }

        export default function Modal({
          id,
          title,
          description,
          content,
          actions,
          size = "md",
          dismissible = true,
          trigger,
          isOpen: controlledIsOpen,
          onClose,
        }: ModalProps) {
          const [internalIsOpen, setInternalIsOpen] = useState(false);
          const isControlled = controlledIsOpen !== undefined;
          const isOpen = isControlled ? controlledIsOpen : internalIsOpen;

          useEffect(() => {
            if (!trigger) return;

            const handleAction = (event: CustomEvent) => {
              if (event.detail.actionId === trigger) {
                if (isControlled && onClose) {
                  onClose(); // Let parent handle
                } else {
                  setInternalIsOpen(true);
                }
              }
            };

            window.addEventListener('namel3ss:action' as any, handleAction);
            return () => window.removeEventListener('namel3ss:action' as any, handleAction);
          }, [trigger, isControlled, onClose]);

          const handleClose = () => {
            if (isControlled && onClose) {
              onClose();
            } else {
              setInternalIsOpen(false);
            }
          };

          const handleBackdropClick = (e: React.MouseEvent) => {
            if (dismissible && e.target === e.currentTarget) {
              handleClose();
            }
          };

          const handleActionClick = (action: ModalAction) => {
            if (action.action) {
              window.dispatchEvent(new CustomEvent('namel3ss:action', {
                detail: { actionId: action.action }
              }));
            }
            if (action.close !== false) {
              handleClose();
            }
          };

          useEffect(() => {
            const handleEscape = (e: KeyboardEvent) => {
              if (e.key === 'Escape' && dismissible && isOpen) {
                handleClose();
              }
            };

            if (isOpen) {
              document.addEventListener('keydown', handleEscape);
              document.body.style.overflow = 'hidden';
            }

            return () => {
              document.removeEventListener('keydown', handleEscape);
              document.body.style.overflow = '';
            };
          }, [isOpen, dismissible]);

          if (!isOpen) return null;

          const sizeClasses = {
            sm: 'max-w-sm',
            md: 'max-w-md',
            lg: 'max-w-lg',
            xl: 'max-w-xl',
            full: 'max-w-full mx-4'
          };

          const variantClasses = {
            default: 'bg-white text-gray-900 border-gray-200 hover:bg-gray-50',
            primary: 'bg-blue-600 text-white hover:bg-blue-700',
            destructive: 'bg-red-600 text-white hover:bg-red-700',
            ghost: 'bg-transparent hover:bg-gray-100',
            link: 'text-blue-600 underline-offset-4 hover:underline'
          };

          return (
            <div
              className="modal-overlay fixed inset-0 bg-black/50 flex items-center justify-center z-50"
              onClick={handleBackdropClick}
              role="dialog"
              aria-modal="true"
              aria-labelledby={`modal-${id}-title`}
              aria-describedby={description ? `modal-${id}-desc` : undefined}
            >
              <div className={`modal-content bg-white rounded-lg shadow-xl ${sizeClasses[size]} w-full`}>
                <div className="modal-header flex items-start justify-between p-6 border-b">
                  <div>
                    <h2 id={`modal-${id}-title`} className="text-xl font-semibold text-gray-900">
                      {title}
                    </h2>
                    {description && (
                      <p id={`modal-${id}-desc`} className="text-sm text-gray-500 mt-1">
                        {description}
                      </p>
                    )}
                  </div>
                  {dismissible && (
                    <button
                      onClick={handleClose}
                      className="text-gray-400 hover:text-gray-600 transition-colors"
                      aria-label="Close modal"
                    >
                      <X size={20} />
                    </button>
                  )}
                </div>
                
                <div className="modal-body p-6">
                  {content}
                </div>
                
                {actions.length > 0 && (
                  <div className="modal-footer flex gap-2 justify-end p-6 border-t bg-gray-50">
                    {actions.map((action, index) => (
                      <button
                        key={index}
                        onClick={() => handleActionClick(action)}
                        className={`px-4 py-2 rounded-md font-medium transition-colors ${
                          variantClasses[action.variant || 'default']
                        }`}
                      >
                        {action.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          );
        }
        """
    ).strip() + "\n"
    write_file(components_dir / "Modal.tsx", content)


def write_toast_component(components_dir: Path) -> None:
    """Generate Toast.tsx notification component."""
    content = textwrap.dedent(
        """
        import { useState, useEffect } from "react";
        import { X, CheckCircle, XCircle, AlertCircle, Info } from "lucide-react";

        export interface ToastProps {
          id: string;
          title: string;
          description?: string;
          variant?: "default" | "success" | "error" | "warning" | "info";
          duration?: number;
          action_label?: string;
          action?: string;
          position?: "top" | "top-right" | "top-left" | "bottom" | "bottom-right" | "bottom-left";
          trigger?: string;
          isVisible?: boolean;
          onDismiss?: () => void;
        }

        export default function Toast({
          id,
          title,
          description,
          variant = "default",
          duration = 3000,
          action_label,
          action,
          position = "top-right",
          trigger,
          isVisible: controlledIsVisible,
          onDismiss,
        }: ToastProps) {
          const [internalIsVisible, setInternalIsVisible] = useState(false);
          const isControlled = controlledIsVisible !== undefined;
          const isVisible = isControlled ? controlledIsVisible : internalIsVisible;

          useEffect(() => {
            if (!trigger) return;

            const handleAction = (event: CustomEvent) => {
              if (event.detail.actionId === trigger) {
                if (isControlled && onDismiss) {
                  // Parent controls visibility
                } else {
                  setInternalIsVisible(true);
                }
              }
            };

            window.addEventListener('namel3ss:action' as any, handleAction);
            return () => window.removeEventListener('namel3ss:action' as any, handleAction);
          }, [trigger, isControlled, onDismiss]);

          useEffect(() => {
            if (isVisible && duration > 0) {
              const timer = setTimeout(() => {
                handleDismiss();
              }, duration);
              return () => clearTimeout(timer);
            }
          }, [isVisible, duration]);

          const handleDismiss = () => {
            if (isControlled && onDismiss) {
              onDismiss();
            } else {
              setInternalIsVisible(false);
            }
          };

          const handleActionClick = () => {
            if (action) {
              window.dispatchEvent(new CustomEvent('namel3ss:action', {
                detail: { actionId: action }
              }));
            }
            handleDismiss();
          };

          if (!isVisible) return null;

          const variantStyles = {
            default: {
              bg: 'bg-white border-gray-200',
              text: 'text-gray-900',
              icon: null
            },
            success: {
              bg: 'bg-green-50 border-green-200',
              text: 'text-green-900',
              icon: <CheckCircle size={20} className="text-green-600" />
            },
            error: {
              bg: 'bg-red-50 border-red-200',
              text: 'text-red-900',
              icon: <XCircle size={20} className="text-red-600" />
            },
            warning: {
              bg: 'bg-yellow-50 border-yellow-200',
              text: 'text-yellow-900',
              icon: <AlertCircle size={20} className="text-yellow-600" />
            },
            info: {
              bg: 'bg-blue-50 border-blue-200',
              text: 'text-blue-900',
              icon: <Info size={20} className="text-blue-600" />
            }
          };

          const positionClasses = {
            'top': 'top-4 left-1/2 -translate-x-1/2',
            'top-right': 'top-4 right-4',
            'top-left': 'top-4 left-4',
            'bottom': 'bottom-4 left-1/2 -translate-x-1/2',
            'bottom-right': 'bottom-4 right-4',
            'bottom-left': 'bottom-4 left-4'
          };

          const style = variantStyles[variant];

          return (
            <div
              className={`toast fixed ${positionClasses[position]} z-50 w-96 max-w-full animate-slide-in`}
              role="alert"
              aria-live="polite"
              aria-atomic="true"
            >
              <div className={`${style.bg} border rounded-lg shadow-lg p-4`}>
                <div className="flex gap-3">
                  {style.icon && <div className="flex-shrink-0">{style.icon}</div>}
                  <div className="flex-1 min-w-0">
                    <h3 className={`font-semibold ${style.text}`}>{title}</h3>
                    {description && (
                      <p className={`text-sm mt-1 ${style.text} opacity-90`}>
                        {description}
                      </p>
                    )}
                    {action_label && (
                      <button
                        onClick={handleActionClick}
                        className={`text-sm font-medium mt-2 ${style.text} underline hover:no-underline`}
                      >
                        {action_label}
                      </button>
                    )}
                  </div>
                  <button
                    onClick={handleDismiss}
                    className={`flex-shrink-0 ${style.text} opacity-50 hover:opacity-100 transition-opacity`}
                    aria-label="Dismiss notification"
                  >
                    <X size={16} />
                  </button>
                </div>
              </div>
            </div>
          );
        }
        """
    ).strip() + "\n"
    write_file(components_dir / "Toast.tsx", content)


def write_all_chrome_components(components_dir: Path) -> None:
    """Write all chrome component files."""
    write_sidebar_component(components_dir)
    write_navbar_component(components_dir)
    write_breadcrumbs_component(components_dir)
    write_command_palette_component(components_dir)
    write_modal_component(components_dir)
    write_toast_component(components_dir)
