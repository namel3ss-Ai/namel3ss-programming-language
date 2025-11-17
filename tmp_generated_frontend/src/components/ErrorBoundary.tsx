import { Component, type ErrorInfo, type ReactNode } from "react";

interface ErrorBoundaryProps {
  fallback?: ReactNode;
  onError?: (error: Error, info: ErrorInfo) => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  message: string;
}

export default class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  public state: ErrorBoundaryState = {
    hasError: false,
    message: "",
  };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return {
      hasError: true,
      message: error?.message ?? "Unexpected error",
    };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    if (typeof this.props.onError === "function") {
      this.props.onError(error, info);
    }
  }

  reset() {
    this.setState({ hasError: false, message: "" });
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }
      return (
        <div role="alert" className="n3-error-boundary" style={{ padding: "1rem", borderRadius: "0.75rem", backgroundColor: "rgba(220,38,38,0.1)", color: "#991b1b" }}>
          <strong>Something went wrong.</strong>
          <div>{this.state.message}</div>
        </div>
      );
    }
    return this.props.children as ReactNode;
  }
}
