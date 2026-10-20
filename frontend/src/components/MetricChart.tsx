import * as echarts from "echarts";
import { useEffect, useMemo, useRef, useState } from "react";
import type { MetricPoint } from "../hooks/useRunStream";

interface MetricChartProps {
  metrics: MetricPoint[];
  theme: "light" | "dark";
}

// Simple moving average for the smoothing toggle [doc 09 §3.3]. Window = 5.
function smooth(values: number[], window: number): number[] {
  if (window < 2) return values;
  const result: number[] = [];
  for (let index = 0; index < values.length; index += 1) {
    const start = Math.max(0, index - window + 1);
    const slice = values.slice(start, index + 1);
    result.push(slice.reduce((sum, value) => sum + value, 0) / slice.length);
  }
  return result;
}

// Live metric chart: loss/accuracy curves + learning rate on a second axis,
// with smoothing and a zoom bar [doc 09 §3.3]. Chart library = ECharts (D8).
export function MetricChart({ metrics, theme }: MetricChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);
  const [smoothing, setSmoothing] = useState(false);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const chart = echarts.init(container);
    chartRef.current = chart;
    const onResize = (): void => chart.resize();
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  // All metric names present, in first-seen order (dynamic model metrics).
  const seriesNames = useMemo(() => {
    const seen: string[] = [];
    for (const point of metrics) {
      if (!seen.includes(point.name)) seen.push(point.name);
    }
    return seen;
  }, [metrics]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    const steps = [...new Set(metrics.map((point) => point.step))].sort((a, b) => a - b);
    const isLr = (name: string): boolean => name === "lr";
    const series = seriesNames.map((name) => {
      const raw = steps.map((step) => {
        const points = metrics.filter((point) => point.name === name && point.step === step);
        return points.length > 0 ? points[points.length - 1].value : NaN;
      });
      const values = smoothing ? smooth(raw.map((value) => (Number.isNaN(value) ? NaN : value)), 5) : raw;
      return {
        name,
        type: "line" as const,
        showSymbol: false,
        smooth: smoothing,
        yAxisIndex: isLr(name) ? 1 : 0,
        connectNulls: false,
        data: steps.map((step, index) => {
          const value = values[index];
          return Number.isNaN(value) ? null : [step, value];
        }),
      };
    });
    const styles = getComputedStyle(document.documentElement);
    chart.setOption({
      animation: false,
      grid: { left: 48, right: 48, top: 40, bottom: 64 },
      legend: { textStyle: { color: styles.getPropertyValue("--th-text-secondary") } },
      tooltip: { trigger: "axis" },
      dataZoom: [
        { type: "inside", xAxisIndex: 0 },
        { type: "slider", xAxisIndex: 0, height: 20, bottom: 8 },
      ],
      xAxis: {
        type: "value",
        name: "step",
        axisLine: { lineStyle: { color: styles.getPropertyValue("--th-border") } },
        axisLabel: { color: styles.getPropertyValue("--th-text-secondary") },
        splitLine: { show: false },
      },
      yAxis: [
        {
          type: "value",
          name: "loss / metric",
          axisLabel: { color: styles.getPropertyValue("--th-text-secondary") },
          splitLine: { lineStyle: { color: styles.getPropertyValue("--th-border") } },
        },
        {
          type: "value",
          name: "lr",
          axisLabel: { color: styles.getPropertyValue("--th-text-secondary") },
          splitLine: { show: false },
        },
      ],
      series,
      color: ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#14b8a6"],
    });
  }, [metrics, theme, seriesNames, smoothing]);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold">Metrics</span>
        <label className="flex items-center gap-2 text-xs text-text-secondary">
          <input
            type="checkbox"
            checked={smoothing}
            onChange={(event) => setSmoothing(event.target.checked)}
          />
          Smooth
        </label>
      </div>
      <div
        ref={containerRef}
        role="img"
        aria-label="Live training metrics: loss, accuracy and learning rate"
        className="h-96 w-full rounded-lg border border-border bg-surface p-2"
      />
    </div>
  );
}
