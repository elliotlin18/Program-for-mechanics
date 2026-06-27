import type { Config } from "tailwindcss";

export default {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        serif: [
          "Iowan Old Style",
          "Palatino Linotype",
          "Palatino",
          "Palladio",
          "Georgia",
          "Times New Roman",
          "serif",
        ],
        sans: [
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica",
          "Arial",
          "sans-serif",
        ],
        mono: [
          "ui-monospace",
          "SF Mono",
          "Menlo",
          "Consolas",
          "Liberation Mono",
          "monospace",
        ],
      },
      fontSize: {
        // Senior-facing surfaces: prefer large base sizes and high contrast.
        senior: ["1.25rem", { lineHeight: "1.8rem" }],
        "senior-lg": ["1.5rem", { lineHeight: "2rem" }],
      },
      colors: {
        // Editorial "Trygg Auktoritet" palette.
        forest: { DEFAULT: "#0B3D2E", 2: "#0E4A38" },
        gold: { DEFAULT: "#B5852A", 2: "#C8A24A" },
        paper: { DEFAULT: "#F6F3EC", 2: "#FBF9F3" },
        ink: "#11201B",
        muted: "#5A6660",
        hair: "#D8D2C4",
        // Kept so existing surfaces (dashboard/senior) stay coherent.
        brand: { DEFAULT: "#0B3D2E", dark: "#0E4A38" },
      },
    },
  },
  plugins: [],
} satisfies Config;
