// Copyright 2020 Aaron Krumins. All Rights Reserved.


#include "MindMaker.h"
#include "Misc/Paths.h"

#define LOCTEXT_NAMESPACE "FMindMakerModule"

void FMindMakerModule::StartupModule()
{
	// This code will execute after your module is loaded into memory; the exact timing is specified in the .uplugin file per-module
	
}

void FMindMakerModule::ShutdownModule()
{
	// This function may be called during shutdown to clean up your module.  For modules that support dynamic reloading,
	// we call this function before unloading the module.
	
}

#undef LOCTEXT_NAMESPACE
	
IMPLEMENT_MODULE(FMindMakerModule, MindMaker)