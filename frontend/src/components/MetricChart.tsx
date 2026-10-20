import * as echarts from "echarts";
import { useEffect, useRef } from "react";
import type { MetricPoint } from "../hooks/useRunStream";

interface MetricChartProps {
  metrics: MetricPoint[];
  theme: "light" | "dark";
}

// Live metric chart: loss curves plus learning rate on a second axis [doc 09 §3.3].
// Chart library = ECharts (D8) [doc 00 §5, doc 09 §4].
export function MetricChart({ metrics, theme }: MetricChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);

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

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    const seriesNames = ["train/loss", "val/loss", "lr"];
    const steps = [...new Set(metrics.map((point) => point.step))].sort((a, b) => a - b);
    const series = seriesNames.map((name) => ({
      name,
      type: "line" as const,
      showSymbol: false,
      yAxisIndex: name === "lr" ? 1 : 0,
      data: steps.map((step) => {
        const points = metrics.filter(
          (point) => point.name === name && point.step === step,
        );
        return points.length > 0 ? [step, points[points.length - 1].value] : null;
      }),
    }));
    const styles = getComputedStyle(document.documentElement);
    chart.setOption({
      animation: false,
      grid: { left: 48, right: 48, top: 40, bottom: 32 },
      legend: { textStyle: { color: styles.getPropertyValue("--th-text-secondary") } },
      tooltip: { trigger: "axis" },
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
          name: "loss",
          axisLabel: { color: styles.getPropertyValue("--th-text-secondary") },
          splitLine: {
            lineStyle: { color: styles.getPropertyValue("--th-border") },
          },
        },
        {
          type: "value",
          name: "lr",
          axisLabel: { color: styles.getPropertyValue("--th-text-secondary") },
          splitLine: { show: false },
        },
      ],
      series,
      color: ["#2563eb", "#f59e0b", "#10b981"],
    });
  }, [metrics, theme]);

  return (
    <div
      ref={containerRef}
      role="img"
      aria-label="Live training metrics: loss and learning rate"
      className="h-80 w-full rounded-lg border border-border bg-surface p-2"
    />
  );
}
