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

	const MAX_CHAR_SIZE = 6
</script>

<div class="mb-5 grid grid-cols-4">
	{#if label}
		<label class="font-main text-ti font-bold text-theme-r" title={label}>{label.length > MAX_CHAR_SIZE ? label.slice(0, MAX_CHAR_SIZE) + "." : label}</label>
	{/if}
	<div class="relative col-span-3 flex flex-row items-center justify-between text-gray-500">
		<span class="absolute -bottom-3 -start-1 text-ti-s font-bold">{min}</span>
		<input
			type="range"
			value={inpVal}
			{min}
			{max}
			{step}
			oninput={(e: any) => updateInpVal(e.target.value)}
			class="w-full accent-theme h-[6px] cursor-pointer {inpStyle}"
		/>
		<span class="absolute -bottom-3 -end-1 text-ti-s font-bold">{max}</span>
		<span
			style="left: {((inpVal - min) / (max - min)) * 100}%;"
			class={'absolute -bottom-[0.8rem] min-w-[1.8rem] translate-x-[-50%] rounded-md bg-theme text-center text-ti-s text-theme-w '}
		>
			{inpVal}
		</span>
	</div>
</div>
