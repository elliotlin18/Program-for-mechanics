import * as React from 'react'
import { cn } from '@/lib/utils'

type Variant = 'default' | 'active' | 'removed' | 'high' | 'medium' | 'low'

const variants: Record<Variant, string> = {
  default: 'bg-muted text-muted-foreground',
  active: 'bg-sky-100 text-sky-800',
  removed: 'bg-emerald-100 text-emerald-800',
  high: 'bg-emerald-100 text-emerald-800',
  medium: 'bg-amber-100 text-amber-800',
  low: 'bg-rose-100 text-rose-800',
}

export function Badge({
  variant = 'default',
  className,
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & { variant?: Variant }) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium',
        variants[variant],
        className,
      )}
      {...props}
    />
  )
}
