"use client";

import {useState, useEffect} from "react";
import {
    ChevronLeft,
    ChevronRight,
    LayoutDashboard,
    Users,
    List,
} from "lucide-react";

interface TabNavigationProps {
    activeTab: "overview" | "player" | "rally";
    onTabChange: (tab: "overview" | "player" | "rally") => void;
    onCollapseChange?: (collapsed: boolean) => void;
}

export default function TabNavigation({
    activeTab,
    onTabChange,
    onCollapseChange,
}: TabNavigationProps) {
    const [isCollapsed, setIsCollapsed] = useState(false);

    useEffect(() => {
        onCollapseChange?.(isCollapsed);
    }, [isCollapsed, onCollapseChange]);

    const tabs = [
        {
            id: "overview" as const,
            label: "Overview",
            description: "Match Summary",
            icon: LayoutDashboard,
        },
        {
            id: "player" as const,
            label: "Player Analysis",
            description: "Positioning & Tactics",
            icon: Users,
        },
        {
            id: "rally" as const,
            label: "Rally-by-Rally",
            description: "Detailed Breakdown",
            icon: List,
        },
    ];

    return (
        <div
            className={`fixed left-0 top-0 h-screen bg-white shadow-xl transition-all duration-300 z-50 ${
                isCollapsed ? "w-16" : "w-64"
            }`}
        >
            {/* Toggle Button */}
            <button
                onClick={() => setIsCollapsed(!isCollapsed)}
                className="absolute -right-3 top-8 bg-dark-red text-white rounded-full p-1.5 shadow-lg hover:bg-dark-red/90 transition-colors"
                aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            >
                {isCollapsed ? (
                    <ChevronRight className="w-4 h-4" />
                ) : (
                    <ChevronLeft className="w-4 h-4" />
                )}
            </button>

            {/* Navigation Items */}
            <div className="flex flex-col pt-8 px-2">
                {tabs.map((tab) => {
                    const Icon = tab.icon;
                    return (
                        <button
                            key={tab.id}
                            onClick={() => onTabChange(tab.id)}
                            className={`flex items-center gap-3 px-4 py-3 rounded-lg mb-2 transition-all ${
                                activeTab === tab.id
                                    ? "bg-dark-red text-white shadow-md"
                                    : "text-black hover:bg-black/5"
                            }`}
                            title={isCollapsed ? tab.label : undefined}
                        >
                            <Icon
                                className={`shrink-0 ${
                                    isCollapsed ? "w-6 h-6" : "w-5 h-5"
                                }`}
                            />
                            {!isCollapsed && (
                                <div className="text-left overflow-hidden">
                                    <div className="text-base font-bold truncate">
                                        {tab.label}
                                    </div>
                                    <div
                                        className={`text-xs truncate ${
                                            activeTab === tab.id
                                                ? "text-white/80"
                                                : "text-black/60"
                                        }`}
                                    >
                                        {tab.description}
                                    </div>
                                </div>
                            )}
                        </button>
                    );
                })}
            </div>
        </div>
    );
}
