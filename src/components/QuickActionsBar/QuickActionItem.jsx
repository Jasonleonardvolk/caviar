import React from "react";

/**
 * QuickActionItem
 * Props:
 * - icon: String or JSX element (agent persona icon)
 * - color: string (accent color)
 * - label: string (short action label)
 * - explanation: string (tooltip/diff preview)
 * - filePath: string (optional - path to the file to modify)
 * - rangeStart: object (optional - start position for the edit)
 * - rangeEnd: object (optional - end position for the edit)
 * - diff: object (optional - contains old and new code)
 * - group: string (optional - suggestion category)
 * - onApply: function (apply handler)
 */
export default function QuickActionItem({ icon, color, label, explanation, onApply }) {
  // Helper function to render icon correctly whether it's a string or JSX
  const renderIcon = (icon) => {
    if (typeof icon === 'string') {
      return <span role="img" aria-label="Agent Icon">{icon}</span>;
    }
    return icon;
  };
  return (
    <div
      className="quick-action-item"
      style={{
        display: "flex",
        alignItems: "center",
        background: color + "11", // subtle background
        border: `1px solid ${color}55`,
        borderRadius: 16,
        padding: "4px 12px",
        marginRight: 8,
        cursor: "pointer",
        position: "relative",
        fontSize: 14,
        transition: "background 0.2s"
      }}
      title={explanation}
      onClick={onApply}
    >
      <span style={{ marginRight: 6 }}>{renderIcon(icon)}</span>
      <span>{label}</span>
    </div>
  );
}
