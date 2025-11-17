import IndexPage from "./pages/index";
import FeedbackPage from "./pages/feedback";
import AdminPage from "./pages/admin";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { ToastProvider } from "./components/Toast";
import ErrorBoundary from "./components/ErrorBoundary";
import ProtectedRoute from "./components/ProtectedRoute";
import { AuthProvider } from "./lib/auth";
import { I18nProvider } from "./lib/i18n";

const DEFAULT_MESSAGES = {
  "en": {
    "loading.generic": "Loading...",
    "errors.generic": "Something went wrong.",
    "errors.forbidden": "You do not have access to view this page.",
    "auth.login": "Sign In",
    "auth.logout": "Sign Out",
    "auth.sessionExpired": "Your session expired. Please sign in again.",
    "navigation.skip": "Skip to content",
    "navigation.main": "Main navigation"
  }
} as const;

export default function App() {
  return (
    <I18nProvider messages={DEFAULT_MESSAGES}>
      <AuthProvider>
        <ToastProvider>
          <BrowserRouter>
            <ErrorBoundary>
              <Routes>
          <Route path="/" element={<IndexPage />} />
          <Route path="/feedback" element={<FeedbackPage />} />
          <Route path="/admin" element={<AdminPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </ErrorBoundary>
          </BrowserRouter>
        </ToastProvider>
      </AuthProvider>
    </I18nProvider>
  );
}
