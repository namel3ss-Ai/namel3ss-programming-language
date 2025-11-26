import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { ToastProvider } from "./components/Toast";
import IndexPage from "./pages/index";
import AnalyticsPage from "./pages/analytics";
import ReportsPage from "./pages/reports";
import ReportsSalesPage from "./pages/reports_sales";
import ProfilePage from "./pages/profile";
import SecurityPage from "./pages/security";

export default function App() {
  return (
    <ToastProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<IndexPage />} />
          <Route path="/analytics" element={<AnalyticsPage />} />
          <Route path="/reports" element={<ReportsPage />} />
          <Route path="/reports/sales" element={<ReportsSalesPage />} />
          <Route path="/profile" element={<ProfilePage />} />
          <Route path="/security" element={<SecurityPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </ToastProvider>
  );
}
