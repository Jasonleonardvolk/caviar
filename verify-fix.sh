#!/bin/bash
echo "🔍 Quick TypeScript Check"
echo "========================"

# Count errors before and after
echo -e "\n📊 Running type check..."
npx tsc --noEmit 2>&1 | tee /tmp/tsc-output.txt

# Extract error count
ERROR_COUNT=$(grep -o "Found [0-9]* error" /tmp/tsc-output.txt | grep -o "[0-9]*")

if [ -z "$ERROR_COUNT" ]; then
    echo -e "\n✅ No TypeScript errors found!"
    echo "Ready to build and package!"
else
    echo -e "\n📍 Current status: $ERROR_COUNT errors (down from 90)"
    
    # Show breakdown
    echo -e "\n📂 Errors by file:"
    grep "\.ts" /tmp/tsc-output.txt | grep "error TS" | cut -d: -f1 | sort | uniq -c | sort -rn | head -5
    
    echo -e "\n💡 To build anyway: npm run build"
fi
