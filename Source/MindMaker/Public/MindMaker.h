// Copyright 2020 Aaron Krumins. All Rights Reserved.


#pragma once

#include "Modules/ModuleManager.h"
#include "Misc/Paths.h"
class FMindMakerModule : public IModuleInterface
{
public:

	/** IModuleInterface implementation */
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;
};
