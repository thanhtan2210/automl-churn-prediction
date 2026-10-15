import { PredictView } from "@/components/predict-view";

export default async function PredictPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <PredictView id={id} />;
}
