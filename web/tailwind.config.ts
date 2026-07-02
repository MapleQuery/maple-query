import type { Config } from "tailwindcss";
import animate from "tailwindcss-animate";

const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: "#F0EEEB",
        "surface-soft": "#E7EBEC",
        "surface-card": "#DDE4E7",
        ink: "#13181B",
        body: "#2D3942",
        muted: "#5C6B75",
        hairline: "#D3DADD",
        coral: "#FD8973",
        "coral-active": "#ED7259",
        navy: "#003A6C",
        polar: "#CCD5DA",
        teal: "#4E8197",
        amber: "#FFBF65",
        success: "#2F7D5B",
        error: "#C2503B",
      },
      fontFamily: {
        // One typeface across the whole app. `display` and `mono` are kept as
        // aliases so we don't have to rewrite every className, but they all
        // resolve to Inter.
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
        display: ["var(--font-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-sans)", "system-ui", "sans-serif"],
      },
      keyframes: {
        rise: {
          from: { opacity: "0", transform: "translateY(6px)" },
          to: { opacity: "1", transform: "none" },
        },
        shimmer: {
          "0%": { backgroundPosition: "200% 0" },
          "100%": { backgroundPosition: "-200% 0" },
        },
        "dot-blink": {
          "0%, 60%, 100%": { opacity: "0.25" },
          "30%": { opacity: "1" },
        },
        "leaf-pulse": {
          "0%, 100%": { transform: "scale(0.78)", opacity: "0.75" },
          "50%": { transform: "scale(1.08)", opacity: "1" },
        },
      },
      animation: {
        rise: "rise 0.3s ease both",
        shimmer: "shimmer 1.4s infinite",
        "dot-blink": "dot-blink 1.2s infinite",
        "leaf-pulse": "leaf-pulse 1.4s ease-in-out infinite",
      },
    },
  },
  plugins: [animate],
};

export default config;
