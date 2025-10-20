interface StatCardProps {
  title: string;
  value: number | string;
  unit?: string;
  icon?: React.ReactNode;
  primary?: boolean;
  subtitle?: string;
}

export default function StatCard({
  title,
  value,
  unit = '',
  icon,
  primary = false,
  subtitle,
}: StatCardProps) {
  return (
    <div
      className={`rounded-xl shadow-lg p-6 ${
        primary ? 'bg-dark-red text-white' : 'bg-white text-black'
      }`}
    >
      <div className="flex items-start justify-between mb-2">
        <h3
          className={`text-sm font-medium ${
            primary ? 'text-white/80' : 'text-black/60'
          }`}
        >
          {title}
        </h3>
        {icon && (
          <div className={primary ? 'text-white' : 'text-dark-red'}>
            {icon}
          </div>
        )}
      </div>
      <div className="flex items-baseline gap-1 mb-1">
        <span className="text-5xl font-bold">{value}</span>
        {unit && (
          <span className={`text-xl ${primary ? 'text-white/60' : 'text-black/40'}`}>
            {unit}
          </span>
        )}
      </div>
      {subtitle && (
        <p className={`text-sm ${primary ? 'text-white/70' : 'text-black/50'}`}>
          {subtitle}
        </p>
      )}
    </div>
  );
}
