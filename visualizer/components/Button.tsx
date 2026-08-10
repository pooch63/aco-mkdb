import type { ButtonHTMLAttributes, ReactNode } from "react";

type Props = ButtonHTMLAttributes<HTMLButtonElement> & {
  children: ReactNode;
  wide?: boolean;
  primary?: boolean;
  danger?: boolean;
  active?: boolean;
};

export function Button({
  children,
  wide,
  primary,
  danger,
  active,
  className = "",
  ...rest
}: Props) {
  const classes = [
    wide ? "wide" : "",
    primary ? "primary" : "",
    danger ? "danger" : "",
    active ? "active" : "",
    className,
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <button className={classes} {...rest}>
      {children}
    </button>
  );
}
