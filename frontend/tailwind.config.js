/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        graphite: {
          900: '#111317',
          800: '#1A1D23',
          700: '#262A33',
          600: '#353B47',
          500: '#485060',
          400: '#647087',
          300: '#8A96AC',
          200: '#BAC3D4',
          100: '#E2E7F0',
          50: '#F4F6F9'
        },
        warmwhite: '#FAF9F6',
        accent: {
          blue: '#2B6CB0',
          cyan: '#00A3C4',
          amber: '#DD6B20',
          crimson: '#C53030',
          emerald: '#2F855A'
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Menlo', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif']
      }
    },
  },
  plugins: [],
}
