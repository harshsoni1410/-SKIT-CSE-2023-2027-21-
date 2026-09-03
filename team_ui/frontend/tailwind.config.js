/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        // dark theme surface colors (see DESIGN.md "Visual style")
        base: '#0b0f14',
        card: '#151b23',
        line: '#232b36',
        accent: {
          DEFAULT: '#2dd4bf', // teal
          soft: '#14b8a6',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
    },
  },
  plugins: [],
}
