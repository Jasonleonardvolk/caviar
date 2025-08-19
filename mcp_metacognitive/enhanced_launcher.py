            print(f"❌ Missing critical dependency: {e}")
            print("💡 Run: pip install psutil requests uvicorn")
            return 1
        
        launcher = EnhancedUnifiedToriLauncher()
        return launcher.launch()
        
    except Exception as e:
        print(f"❌ Critical startup failure: {e}")
        print(f"📋 Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
