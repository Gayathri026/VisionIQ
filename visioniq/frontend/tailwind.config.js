/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Command Center Dark Palette (Design Tokens)
        command: {
          bg: '#0B1020',
          bgSecondary: '#10182B',
          panel: '#151F35',
          panelElevated: '#1B2942',
          panelHover: '#223252',
          border: '#2B3A54',
          borderLight: '#3A4D6F',
        },
        navy: {
          950: '#0B1020',
          900: '#10182B',
          850: '#151F35',
          800: '#1B2942',
          750: '#223252',
          700: '#2B3A54',
          600: '#3D5174',
          500: '#546E9B',
          400: '#758FB8',
          300: '#A0B4D4',
          200: '#C8D6EC',
          100: '#E6EEF8',
        },
        accent: {
          cyan: '#43D9F5',
          cyanGlow: 'rgba(67, 217, 245, 0.25)',
          indigo: '#818CF8',
          blue: '#3B82F6',
          amber: '#FBBF24',
          coral: '#FB7185',
          emerald: '#34D399',
          teal: '#2DD4BF',
        },
        status: {
          success: '#34D399',
          warning: '#FBBF24',
          critical: '#FB7185',
          info: '#43D9F5',
          neutral: '#8998AF',
        },
        content: {
          primary: '#F1F5F9',
          secondary: '#B4C0D4',
          muted: '#8998AF',
        },
        // Backward-compatibility aliases with elevated contrast
        graphite: {
          950: '#0B1020',
          900: '#10182B',
          850: '#151F35',
          800: '#1B2942',
          700: '#2B3A54',
          600: '#3D5174',
          500: '#647D9E',
          400: '#8998AF',
          300: '#B4C0D4',
          200: '#CBD5E1',
          100: '#E2E8F0',
          50: '#F8FAFC',
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Menlo', 'monospace'],
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      boxShadow: {
        'command-sm': '0 2px 8px rgba(0, 0, 0, 0.35)',
        'command-md': '0 4px 16px rgba(0, 0, 0, 0.45)',
        'command-lg': '0 8px 30px rgba(0, 0, 0, 0.55)',
        'cyan-glow': '0 0 20px rgba(67, 217, 245, 0.2)',
        'cyan-glow-lg': '0 0 35px rgba(67, 217, 245, 0.35)',
      }
    },
  },
  plugins: [],
}
