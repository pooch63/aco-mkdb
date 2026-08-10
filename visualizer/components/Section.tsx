import type { ReactNode } from "react";

export function Section({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <div className="section">
      <h2>{title}</h2>
      {children}
    </div>
  );
}
