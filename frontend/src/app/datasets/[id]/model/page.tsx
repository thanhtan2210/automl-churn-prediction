import { ModelView } from "@/components/model-view";

export default async function ModelPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <ModelView id={id} />;
}
