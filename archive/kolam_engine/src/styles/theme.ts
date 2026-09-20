// theme.ts — chalkboard topology color system

export const theme = {
  bg: {
    app:   '#274C43',
    panel: '#17332D',
    card:  '#0B1F1B',
  },
  text: {
    primary:   '#E5E7EB',
    secondary: '#9CA3AF',
  },
  chalk: {
    main:      '#F9FAFB',  // kolam curve
    highlight: '#FACC15',  // hover / step
    alt:       '#FB7185',  // secondary path
    guide:     '#22D3EE',  // axes / helpers
    success:   '#6EE7B7',  // valid region
  },
  grid:  '#1F2933',
  dots:  '#9CA3AF',
} as const
