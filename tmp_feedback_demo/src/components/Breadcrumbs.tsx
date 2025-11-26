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
