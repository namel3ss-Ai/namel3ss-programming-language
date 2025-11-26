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
