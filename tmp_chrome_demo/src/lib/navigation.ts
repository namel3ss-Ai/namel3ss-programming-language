export interface NavLink {
          label: string;
          path: string;
        }

        export const NAV_LINKS: NavLink[] = [
  {
    "label": "Dashboard",
    "path": "/"
  },
  {
    "label": "Analytics",
    "path": "/analytics"
  },
  {
    "label": "Reports",
    "path": "/reports"
  },
  {
    "label": "Sales Report",
    "path": "/reports/sales"
  },
  {
    "label": "Profile",
    "path": "/profile"
  },
  {
    "label": "Security",
    "path": "/security"
  }
] as const;
