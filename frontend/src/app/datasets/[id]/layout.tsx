import { DatasetNav } from "@/components/dataset-nav";

export default async function DatasetLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return (
    <>
      <DatasetNav id={id} />
      {children}
    </>
  );
}
