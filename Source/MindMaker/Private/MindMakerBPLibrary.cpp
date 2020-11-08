// Copyright 2020 Aaron Krumins. All Rights Reserved.


#include "MindMakerBPLibrary.h"
#include "MindMaker.h"
#include "Misc/Paths.h"

UMindMakerBPLibrary::UMindMakerBPLibrary(const FObjectInitializer& ObjectInitializer)
: Super(ObjectInitializer)
{

}

void UMindMakerBPLibrary::MindMakerWindows()
{
	FString cmdToRun = FPaths::ProjectContentDir() + "MindMaker\\dist\\mindmaker\\mindmaker.exe";
	FPlatformProcess::CreateProc(*cmdToRun, nullptr, false, true, false, nullptr, 1, nullptr, nullptr);
    
}

