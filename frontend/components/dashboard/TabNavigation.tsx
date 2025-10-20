interface TabNavigationProps {
  activeTab: 'overview' | 'player' | 'rally';
  onTabChange: (tab: 'overview' | 'player' | 'rally') => void;
}

export default function TabNavigation({ activeTab, onTabChange }: TabNavigationProps) {
  const tabs = [
    { id: 'overview' as const, label: 'Overview', description: 'Match Summary' },
    { id: 'player' as const, label: 'Player Analysis', description: 'Positioning & Tactics' },
    { id: 'rally' as const, label: 'Rally-by-Rally', description: 'Detailed Breakdown' },
  ];

  return (
    <div className="bg-white rounded-xl shadow-lg p-2 mb-8">
      <div className="flex gap-2">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`flex-1 px-6 py-4 rounded-lg transition-all ${
              activeTab === tab.id
                ? 'bg-dark-red text-white shadow-md'
                : 'bg-transparent text-black hover:bg-black/5'
            }`}
          >
            <div className="text-lg font-bold">{tab.label}</div>
            <div
              className={`text-sm ${
                activeTab === tab.id ? 'text-white/80' : 'text-black/60'
              }`}
            >
              {tab.description}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
