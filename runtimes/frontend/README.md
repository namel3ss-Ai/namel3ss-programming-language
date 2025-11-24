# Namel3ss Frontend Runtime

Static site and React app generation runtime for the Namel3ss programming language.

## Overview

This package adapts Namel3ss intermediate representation (IR) to frontend applications. It supports generating static HTML/CSS/JS sites and React-based single-page applications, demonstrating that **Namel3ss is a language that targets multiple runtimes**, not just backends.

## Installation

```bash
pip install namel3ss-runtime-frontend
```

Or install with optional features:

```bash
# With all frontend features
pip install namel3ss-runtime-frontend[full]

# Development dependencies
pip install namel3ss-runtime-frontend[dev]
```

## Usage

### Generate Static Site from IR

```python
from namel3ss import Parser, build_frontend_ir
from namel3ss_runtime_frontend import generate_static_site

# Parse .ai source
source = '''
app "MyApp" connects to postgres "DB".

page "Home" at "/" {
    title: "Welcome"
    description: "My Namel3ss app"
}
'''

parser = Parser(source)
module = parser.parse()
app = module.body[0]

# Build runtime-agnostic IR
ir = build_frontend_ir(app)

# Generate static site
generate_static_site(ir, output_dir="frontend/")
```

### Generate React App from IR

```python
from namel3ss_runtime_frontend import generate_react_app

# Generate React SPA
generate_react_app(ir, output_dir="frontend/")
```

### Build and Serve

```bash
# Static site
cd frontend
python -m http.server 8080

# React app
cd frontend
npm install
npm run dev
```

## What Gets Generated

### Static Site Output

```
frontend/
├── index.html                 # Entry point
├── pages/
│   ├── home.html
│   ├── about.html
│   └── contact.html
├── styles/
│   ├── main.css
│   ├── components.css
│   └── themes.css
├── scripts/
│   ├── main.js
│   ├── api-client.js         # Backend API calls
│   └── components.js         # Reusable components
└── assets/
    ├── images/
    └── fonts/
```

### React App Output

```
frontend/
├── package.json               # Dependencies
├── tsconfig.json              # TypeScript config (if enabled)
├── vite.config.js             # Build configuration
├── index.html
├── src/
│   ├── main.tsx               # App entry point
│   ├── App.tsx                # Root component
│   ├── pages/
│   │   ├── HomePage.tsx
│   │   ├── AboutPage.tsx
│   │   └── ContactPage.tsx
│   ├── components/
│   │   ├── Layout.tsx
│   │   ├── Navigation.tsx
│   │   └── PromptCard.tsx
│   ├── api/
│   │   ├── client.ts          # API client
│   │   └── types.ts           # TypeScript types
│   ├── hooks/
│   │   └── usePrompt.ts       # Custom hooks
│   └── styles/
│       ├── index.css
│       └── theme.css
└── public/
    └── assets/
```

## Features

### ✅ Static Site Generation
- Semantic HTML5
- Responsive CSS (mobile-first)
- Vanilla JavaScript (no dependencies)
- Progressive enhancement
- SEO-optimized

### ✅ React App Generation
- TypeScript support
- React Router for navigation
- Custom hooks for API calls
- Component-based architecture
- Vite build tooling

### ✅ API Integration
- Automatic client generation for backend endpoints
- Type-safe API calls (TypeScript)
- Error handling
- Loading states
- Caching support

### ✅ Theming & Styling
- CSS variables for theming
- Dark mode support
- Responsive design
- Accessibility (WCAG 2.1 AA)

### ✅ Forms & Validation
- Generated forms from prompt/agent schemas
- Client-side validation
- Error messages
- Submit handlers

## Configuration

### Static Site Options

```python
from namel3ss_runtime_frontend import generate_static_site

generate_static_site(
    ir=frontend_ir,
    output_dir="frontend/",
    theme="modern",              # modern, classic, minimal
    enable_dark_mode=True,       # Dark mode toggle
    analytics=None,              # Google Analytics ID
    meta_tags={
        "og:image": "/assets/preview.png",
        "twitter:card": "summary_large_image",
    },
)
```

### React App Options

```python
from namel3ss_runtime_frontend import generate_react_app

generate_react_app(
    ir=frontend_ir,
    output_dir="frontend/",
    typescript=True,             # Use TypeScript
    router="react-router",       # Routing library
    state_management="zustand",  # State management
    ui_library=None,             # Material-UI, Chakra, etc.
    api_base_url="http://localhost:8000",
)
```

## Architecture

### IR → Frontend Mapping

| IR Component | Static Site | React App |
|--------------|-------------|-----------|
| `PageSpec` | HTML page | React component + route |
| `FrameSpec` | HTML section | React component |
| `PromptSpec` | Form + JS handler | `usePrompt()` hook |
| `AgentSpec` | Multi-step form | Wizard component |
| `TypeSpec` | HTML form fields | TypeScript interface |
| `RouteSpec` | Page link | `<Link>` component |

### Component Hierarchy (React)

```
<App>
  ├─ <Router>
  │   ├─ <Layout>
  │   │   ├─ <Navigation>
  │   │   ├─ <Outlet>
  │   │   │   ├─ <HomePage>
  │   │   │   ├─ <AboutPage>
  │   │   │   └─ <PromptPage>
  │   │   │       └─ <PromptForm>
  │   │   └─ <Footer>
  │   └─ ...
  └─ <Providers>
      ├─ <APIProvider>
      ├─ <ThemeProvider>
      └─ <ErrorBoundary>
```

## Advanced Usage

### Custom Components (React)

```typescript
// frontend/src/components/custom/MyComponent.tsx
import React from 'react';
import { usePrompt } from '../../hooks/usePrompt';

export const MyComponent: React.FC = () => {
  const { execute, loading, result } = usePrompt('MyPrompt');

  const handleSubmit = async (input: string) => {
    await execute({ text: input });
  };

  return (
    <div>
      {/* Your custom UI */}
    </div>
  );
};
```

### Custom Styles

```css
/* frontend/src/styles/custom.css */
:root {
  --primary-color: #007bff;
  --secondary-color: #6c757d;
  --background: #ffffff;
}

[data-theme="dark"] {
  --background: #1a1a1a;
}
```

### API Client Customization

```typescript
// frontend/src/api/custom-client.ts
import { apiClient } from './client';

// Add custom interceptors
apiClient.interceptors.request.use((config) => {
  // Add auth token
  config.headers.Authorization = `Bearer ${getToken()}`;
  return config;
});
```

## Development

### Static Site Development

```bash
cd frontend
python -m http.server 8080

# Or use a live reload server
npx live-server .
```

### React App Development

```bash
cd frontend

# Install dependencies
npm install

# Start dev server (with hot reload)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Deployment

### Static Site

Deploy to any static hosting:

```bash
# Netlify
netlify deploy --dir=frontend --prod

# Vercel
vercel --prod frontend

# GitHub Pages
cp -r frontend/* docs/
git add docs && git commit -m "Deploy" && git push
```

### React App

```bash
# Build
cd frontend
npm run build

# Deploy dist/ folder to hosting
netlify deploy --dir=dist --prod
```

## Examples

### Simple Landing Page

```namel3ss
app "LandingPage".

page "Home" at "/" {
    title: "Welcome to Namel3ss"
    description: "Build AI apps fast"
}

frame "Hero" {
    heading: "The AI Programming Language"
    subheading: "Compile to multiple runtimes"
}
```

**Generated:** Static HTML with hero section, responsive layout.

### Interactive Prompt Interface

```namel3ss
app "PromptTester" connects to postgres "DB".

prompt "Generate" {
    model: "gpt-4o-mini"
    template: "Create {{content_type}} about {{topic}}."
}

page "Playground" at "/playground" {
    title: "Prompt Playground"
    has_form: true
}
```

**Generated:** React form component with real-time API calls and result display.

## Relationship to Core

The frontend runtime **depends on** the Namel3ss language core:

```
namel3ss (core)              ← Provides IR types, parser
    ↑
    | imports
    |
namel3ss-runtime-frontend    ← Consumes IR, generates frontends
```

**Dependency rules:**
- ✅ Frontend runtime can import from `namel3ss` core
- ❌ Core CANNOT import from frontend runtime
- ✅ Frontend runtime is independent of other runtimes

## Alternative Runtimes

Namel3ss supports multiple runtime targets:

- **namel3ss-runtime-frontend** (this package) - Static sites, React apps
- **namel3ss-runtime-http** - FastAPI/HTTP backends
- **namel3ss-runtime-deploy** - Docker, Kubernetes, cloud platforms
- **Custom runtimes** - Build your own! (Vue, Svelte, mobile, etc.)

See [docs/RUNTIME_GUIDE.md](../../docs/RUNTIME_GUIDE.md) for creating custom runtimes.

## Testing

### Run Tests

```bash
cd runtimes/frontend
pytest
```

### Visual Testing

```bash
# Install dependencies
npm install -g playwright

# Run visual tests
playwright test
```

## License

MIT License - see LICENSE file for details.

## Links

- **Repository:** https://github.com/SsebowaDisan/namel3ss-programming-language
- **Documentation:** https://github.com/SsebowaDisan/namel3ss-programming-language/tree/main/runtimes/frontend
- **Issues:** https://github.com/SsebowaDisan/namel3ss-programming-language/issues
- **Language Core:** https://github.com/SsebowaDisan/namel3ss-programming-language/tree/main/namel3ss
