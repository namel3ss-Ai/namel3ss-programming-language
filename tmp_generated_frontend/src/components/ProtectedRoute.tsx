import type { PropsWithChildren } from "react";
import { Navigate, useLocation } from "react-router-dom";
import ErrorBanner from "./ErrorBanner";
import { PageLoader } from "./LoadingState";
import { useAuth } from "../lib/auth";
import { useI18n } from "../lib/i18n";

interface ProtectedRouteProps {
  requiresAuth?: boolean;
  allowedRoles?: readonly string[];
  redirectTo?: string | null;
  slug?: string;
  isLogin?: boolean;
}

export default function ProtectedRoute({
  requiresAuth = false,
  allowedRoles,
  redirectTo,
  slug,
  isLogin,
  children,
}: PropsWithChildren<ProtectedRouteProps>) {
  const location = useLocation();
  const { status, user, error } = useAuth();
  const { t } = useI18n();

  const authenticated = status === "authenticated";
  const pending = status === "loading";
  const target = redirectTo ?? "/";

  if (isLogin && authenticated) {
    const from = (location.state as { from?: string })?.from;
    return <Navigate to={from ?? target} replace />;
  }

  if (requiresAuth) {
    if (pending) {
      return <PageLoader />;
    }
    if (!authenticated) {
      return <Navigate to="/login" replace state={{ from: location.pathname + location.search }} />;
    }
    if (allowedRoles && allowedRoles.length) {
      const roles = new Set(
        Array.isArray(user?.roles)
          ? (user?.roles as Array<string | number>).map((role) => String(role))
          : [],
      );
      const hasRole = allowedRoles.some((role) => roles.has(String(role)));
      if (!hasRole) {
        return <ErrorBanner errors={[t("errors.forbidden")]} tone="warning" />;
      }
    }
  }

  if (error && !authenticated) {
    return <ErrorBanner errors={[error]} tone="warning" />;
  }

  return <>{children}</>;
}
