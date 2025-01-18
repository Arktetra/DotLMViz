<script lang="ts">
	const {
		min = 0,
		max = 1,
		step = 0.1,
		label = '',
		changeEventCb = null,
		inpStyle = ''
	} = $props();

	let inpVal: number = $state(
		step % 1 === 0 ? Math.floor((max - min) / 2 + min) : (max - min) / 2 + min
	);

	const updateInpVal = (v: number) => {
		if (changeEventCb) changeEventCb(v);
		inpVal = v;
	};
</script>

<div class="mb-5 flex w-full flex-col">
	{#if label}
		<label class="font-main text-sm font-bold text-theme-r">{label}</label>
	{/if}
	<div class="relative flex w-full flex-row items-center justify-between text-gray-500">
		<span class="absolute -bottom-4 -start-1 text-ti font-bold">{min}</span>
		<input
			type="range"
			value={inpVal}
			{min}
			{max}
			{step}
			oninput={(e: any) => updateInpVal(e.target.value)}
			class="w-full accent-theme {inpStyle}"
		/>
		<span class="absolute -bottom-4 -end-1 text-ti font-bold">{max}</span>
		<span
			style="left: {((inpVal - min) / (max - min)) * 100}%;"
			class={'absolute -bottom-4 min-w-[1.8rem] translate-x-[-50%] rounded-md bg-theme text-center text-ti-s text-theme-w '}
		>
			{inpVal}
		</span>
	</div>
</div>
