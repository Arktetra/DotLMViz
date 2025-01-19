<script lang="ts">
	import { page } from '$app/state';
	console.log(page.params);
	const title = page.params.title.split('-').join(' ');
</script>

<div class="flex flex-col items-center justify-evenly">
	<span class="text-md uppercase">Readings</span>
	<div
		class="my-10 min-h-[80vh] w-[80%] rounded-sm border-2 border-theme-g p-10 text-justify text-lg tracking-wider shadow-lg"
	>
		<h1 class="mb-10 text-center text-4xl font-extrabold uppercase text-theme">{title}</h1>
		<p class="h-full w-full">
			This is the page containing the information about <b>{title}</b>. Let's consider
			<b>{title}</b> is used in this page to do this... and that...
		</p>
	</div>
</div>
