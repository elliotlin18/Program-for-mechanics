'use client'

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { WeekPoint } from '@/lib/stats'
import { formatSek } from '@/lib/utils'

export function PriceChart({ data }: { data: WeekPoint[] }) {
  const hasData = data.some((point) => point.median !== null)

  if (!hasData) {
    return (
      <p className="py-12 text-center text-sm text-muted-foreground">
        Ingen prishistorik ännu för den här modellen.
      </p>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(214 20% 90%)" vertical={false} />
        <XAxis dataKey="week" tickLine={false} axisLine={false} fontSize={12} />
        <YAxis
          tickLine={false}
          axisLine={false}
          fontSize={12}
          width={70}
          tickFormatter={(value: number) => `${Math.round(value / 1000)}k`}
          domain={['auto', 'auto']}
        />
        <Tooltip
          formatter={(value: number) => [formatSek(value), 'Medianpris']}
          labelFormatter={(label: string) => `Vecka från ${label}`}
        />
        <Line
          type="monotone"
          dataKey="median"
          stroke="hsl(201 90% 35%)"
          strokeWidth={2}
          dot={{ r: 3 }}
          connectNulls
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
