export interface NavLink {
          label: string;
          path: string;
        }

        export const NAV_LINKS: NavLink[] = [
  {
    "label": "Feedback Demo",
    "path": "/"
  },
  {
    "label": "User Settings",
    "path": "/settings"
  },
  {
    "label": "Data Management",
    "path": "/data"
  }
] as const;
