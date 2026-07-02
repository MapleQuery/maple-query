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
        display: ["var(--font-display)", "Georgia", "serif"],
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
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
      },
      animation: {
        rise: "rise 0.3s ease both",
        shimmer: "shimmer 1.4s infinite",
        "dot-blink": "dot-blink 1.2s infinite",
      },
    },
  },
  plugins: [animate],
};

export default config;
