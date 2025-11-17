import type { PropsWithChildren } from "react";
import { Link, useLocation } from "react-router-dom";
import { useAuth } from "../lib/auth";
import { useI18n } from "../lib/i18n";
import { NavLink } from "../lib/navigation";

interface LayoutProps {
  title: string;
  description?: string | null;
  navLinks: readonly NavLink[];
}

export default function Layout({ title, description, navLinks, children }: PropsWithChildren<LayoutProps>) {
  const location = useLocation();
  const { status, user } = useAuth();
  const { t } = useI18n();

  const authenticated = status === "authenticated";
  const roles = new Set(
    Array.isArray(user?.roles)
      ? (user?.roles as Array<string | number>).map((role) => String(role))
      : [],
  );

  const filteredNav = navLinks.filter((link) => {
    if (link.hideWhenAuthenticated && authenticated) {
      return false;
    }
    if (link.onlyWhenAuthenticated && !authenticated) {
      return false;
    }
    if (link.requiresAuth && !authenticated) {
      return false;
    }
    if (link.allowedRoles && link.allowedRoles.length) {
      if (!authenticated) {
        return false;
      }
      return link.allowedRoles.some((role) => roles.has(String(role)));
    }
    return true;
  });

  return (
    <div className="n3-app">
      <a href="#n3-main" className="n3-skip-link">
        {t("navigation.skip")}
      </a>
      <header className="n3-header">
        <div className="n3-header__inner">
          <h1 className="n3-header__title">{title}</h1>
          {description ? (
            <p className="n3-header__description">{description}</p>
          ) : null}
          <nav className="n3-nav" aria-label={t("navigation.main")}>
            {filteredNav.map((link) => {
              const isActive = location.pathname === link.path;
              return (
                <Link
                  key={link.path}
                  to={link.path}
                  className={isActive ? "active" : undefined}
                  aria-current={isActive ? "page" : undefined}
                >
                  {link.label}
                </Link>
              );
            })}
          </nav>
        </div>
      </header>
      <main id="n3-main" className="n3-main" tabIndex={-1}>
        {children}
      </main>
    </div>
  );
}
