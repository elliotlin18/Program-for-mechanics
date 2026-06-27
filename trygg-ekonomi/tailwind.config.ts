import type { Config } from "tailwindcss";

export default {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      // Senior-facing surfaces: prefer large base sizes and high contrast.
      fontSize: {
        senior: ["1.25rem", { lineHeight: "1.8rem" }],
        "senior-lg": ["1.5rem", { lineHeight: "2rem" }],
      },
      colors: {
        brand: {
          DEFAULT: "#0f766e", // teal-700 — calm, trustworthy, not alarming
          dark: "#115e59",
        },
      },
    },
  },
  plugins: [],
} satisfies Config;
