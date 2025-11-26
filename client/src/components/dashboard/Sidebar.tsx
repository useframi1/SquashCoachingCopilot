"use client";

import {useState} from "react";
import Link from "next/link";
import {usePathname, useRouter} from "next/navigation";
import {
    LayoutDashboard,
    TrendingUp,
    Move,
    Activity,
    ChevronLeft,
    X,
    Plus,
} from "lucide-react";
import {cn} from "@/lib/utils/cn";
import {Button} from "@/components/ui/button";
import {Separator} from "@/components/ui/separator";

interface SidebarProps {
    videoId: string;
    mobileOpen?: boolean;
    onMobileToggle?: () => void;
}

const navItems = [
    {
        href: "overview",
        icon: LayoutDashboard,
        label: "Overview",
    },
    {
        href: "performance",
        icon: TrendingUp,
        label: "Performance & Shots",
    },
    {
        href: "movement",
        icon: Move,
        label: "Movement & Positioning",
    },
    {
        href: "rallies",
        icon: Activity,
        label: "Rally Analysis",
    },
];

/**
 * Collapsible sidebar navigation for dashboard tabs
 * Responsive: Desktop = collapsible sidebar, Mobile = drawer overlay
 */
export function Sidebar({
    videoId,
    mobileOpen: externalMobileOpen,
    onMobileToggle,
}: SidebarProps) {
    const [collapsed, setCollapsed] = useState(false);
    const [internalMobileOpen, setInternalMobileOpen] = useState(false);
    const pathname = usePathname();
    const router = useRouter();

    // Use external control if provided, otherwise use internal state
    const mobileOpen = externalMobileOpen ?? internalMobileOpen;

    const handleNewAnalysis = () => {
        router.push('/');
        onMobileToggle?.() ?? setInternalMobileOpen(false);
    };

    const SidebarContent = () => (
        <div className="h-full flex flex-col">
            {/* Header */}
            <div className="h-16 flex items-center justify-between px-4 border-b border-gray-200 shrink-0">
                {!collapsed && (
                    <h2 className="text-lg font-semibold text-gray-900">
                        Dashboard
                    </h2>
                )}
                <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setCollapsed(!collapsed)}
                    className="hidden md:flex"
                    aria-label={
                        collapsed ? "Expand sidebar" : "Collapse sidebar"
                    }
                >
                    <ChevronLeft
                        className={cn(
                            "w-5 h-5 transition-transform",
                            collapsed && "rotate-180"
                        )}
                    />
                </Button>
                <Button
                    variant="ghost"
                    size="icon"
                    onClick={() =>
                        onMobileToggle?.() ?? setInternalMobileOpen(false)
                    }
                    className="md:hidden"
                    aria-label="Close menu"
                >
                    <X className="w-5 h-5" />
                </Button>
            </div>

            {/* Navigation */}
            <nav className="p-2 flex flex-col flex-1 overflow-hidden">
                <div className="space-y-1">
                    {navItems.map(({href, icon: Icon, label}) => {
                        const fullPath = `/dashboard/${videoId}/${href}`;
                        const isActive = pathname === fullPath;

                        return (
                            <Link
                                key={href}
                                href={fullPath}
                                onClick={() =>
                                    onMobileToggle?.() ??
                                    setInternalMobileOpen(false)
                                }
                            >
                                <div
                                    className={cn(
                                        "flex items-center gap-3 px-4 py-3 rounded-lg transition-colors",
                                        isActive
                                            ? "bg-red-700 text-white"
                                            : "text-gray-700 hover:bg-gray-100",
                                        collapsed && "justify-center"
                                    )}
                                >
                                    <Icon className="w-5 h-5 shrink-0" />
                                    {!collapsed && (
                                        <span className="font-medium">{label}</span>
                                    )}
                                </div>
                            </Link>
                        );
                    })}
                </div>

                {/* New Analysis Button */}
                <div className="mt-auto pt-2">
                    <Separator className="mb-2" />
                    <Button
                        onClick={handleNewAnalysis}
                        className={cn(
                            "w-full bg-red-700 hover:bg-red-800",
                            collapsed && "px-0"
                        )}
                        aria-label="New Analysis"
                    >
                        <Plus className="w-5 h-5 shrink-0" />
                        {!collapsed && (
                            <span className="ml-2 font-medium">New Analysis</span>
                        )}
                    </Button>
                </div>
            </nav>
        </div>
    );

    return (
        <>
            {/* Desktop Sidebar */}
            <aside
                className={cn(
                    "hidden md:block h-screen bg-white border-r border-gray-200 transition-all duration-300",
                    collapsed ? "w-20" : "w-64"
                )}
            >
                <SidebarContent />
            </aside>

            {/* Mobile Overlay */}
            {mobileOpen && (
                <div
                    className="fixed inset-0 bg-black bg-opacity-50 z-40 md:hidden"
                    onClick={() =>
                        onMobileToggle?.() ?? setInternalMobileOpen(false)
                    }
                />
            )}

            {/* Mobile Drawer */}
            <aside
                className={cn(
                    "fixed top-0 left-0 h-screen w-64 bg-white border-r border-gray-200 z-50 transition-transform duration-300 md:hidden",
                    mobileOpen ? "translate-x-0" : "-translate-x-full"
                )}
            >
                <SidebarContent />
            </aside>
        </>
    );
}
