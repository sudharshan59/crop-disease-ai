interface SeverityBadgeProps {
  severity: string;
}

const severityConfig: Record<
  string,
  { bg: string; text: string; ring: string; dot: string; label: string }
> = {
  healthy: {
    bg: "bg-green-50",
    text: "text-green-700",
    ring: "ring-green-600/20",
    dot: "bg-green-500",
    label: "Healthy",
  },
  mild: {
    bg: "bg-yellow-50",
    text: "text-yellow-700",
    ring: "ring-yellow-600/20",
    dot: "bg-yellow-500",
    label: "Mild",
  },
  moderate: {
    bg: "bg-orange-50",
    text: "text-orange-700",
    ring: "ring-orange-600/20",
    dot: "bg-orange-500",
    label: "Moderate",
  },
  severe: {
    bg: "bg-red-50",
    text: "text-red-700",
    ring: "ring-red-600/20",
    dot: "bg-red-500",
    label: "Severe",
  },
};

export default function SeverityBadge({ severity }: SeverityBadgeProps) {
  const config = severityConfig[severity.toLowerCase()] ?? severityConfig.mild;

  return (
    <span
      className={`
        inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold
        ring-1 ring-inset ${config.bg} ${config.text} ${config.ring}
      `}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${config.dot}`} />
      {config.label}
    </span>
  );
}
