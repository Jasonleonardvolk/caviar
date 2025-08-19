        # Calculate cluster centers
        if premise_projections:
            premise_center = np.mean([p.real for p in premise_projections], axis=0)
            ax.scatter(
                premise_center[0], premise_center[1], premise_center[2],
                s=120, c='darkblue', marker='X', alpha=0.8, edgecolors='white',
                label='Premise Center'
            )
        
        if conclusion_projections:
            conclusion_center = np.mean([p.real for p in conclusion_projections], axis=0)
            ax.scatter(
                conclusion_center[0], conclusion_center[1], conclusion_center[2],
                s=120, c='darkred', marker='X', alpha=0.8, edgecolors='white',
                label='Conclusion Center'
            )
        
        ax.set_xlabel('Eigenfunction 1')
        ax.set_ylabel('Eigenfunction 2')
        ax.set_zlabel('Eigenfunction 3')
        ax.legend()
        plt.title('Eigenfunction Alignment Analysis')
        plt.tight_layout()
        return fig
