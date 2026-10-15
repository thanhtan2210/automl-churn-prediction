"use client";

import { CircleHelp } from "lucide-react";

import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";

export interface Faq {
  q: string;
  a: React.ReactNode;
}

/** "How to read this page" button opening a side panel of short questions and answers. */
export function HowToRead({ items, intro }: { items: Faq[]; intro?: string }) {
  return (
    <Sheet>
      <SheetTrigger render={<Button variant="outline" size="sm" />}>
        <CircleHelp aria-hidden />
        How to read this page
      </SheetTrigger>
      <SheetContent className="overflow-y-auto data-[side=right]:w-full data-[side=right]:sm:max-w-md">
        <SheetHeader>
          <SheetTitle>How to read this page</SheetTitle>
          {intro && <SheetDescription>{intro}</SheetDescription>}
        </SheetHeader>
        <Accordion className="px-4 pb-6" defaultValue={[items[0]?.q]}>
          {items.map((item) => (
            <AccordionItem key={item.q} value={item.q}>
              <AccordionTrigger>{item.q}</AccordionTrigger>
              <AccordionContent className="text-muted-foreground">{item.a}</AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </SheetContent>
    </Sheet>
  );
}
