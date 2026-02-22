import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Agriculture-themed green palette
        leaf: {
          50: "#f0fdf4",
          100: "#dcfce7",
          200: "#bbf7d0",
          300: "#86efac",
          400: "#4ade80",
          500: "#22c55e",
          600: "#16a34a",
          700: "#15803d",
          800: "#166534",
          900: "#14532d",
        },
        soil: {
          50: "#faf5f0",
          100: "#f0e6d3",
          200: "#e0cba6",
          300: "#c9a76e",
          400: "#b5894a",
          500: "#a67639",
          600: "#8c5e2e",
          700: "#724a27",
          800: "#5e3d25",
          900: "#4e3322",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "spin-slow": "spin 2s linear infinite",
      },
    },
  },
  plugins: [],
};

export default config;
