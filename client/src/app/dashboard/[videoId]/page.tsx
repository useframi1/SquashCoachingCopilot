import { redirect } from 'next/navigation';

interface DashboardPageProps {
  params: { videoId: string };
}

/**
 * Dashboard root - redirects to overview tab
 */
export default function DashboardPage({ params }: DashboardPageProps) {
  redirect(`/dashboard/${params.videoId}/overview`);
}
