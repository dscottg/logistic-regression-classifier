scalaVersion := "2.11.1"

libraryDependencies  ++= Seq(
            "org.scalatest" % "scalatest_2.11" % "2.2.1" % "test",
            // other dependencies here
            "org.scalanlp" %% "breeze" % "0.10",
            // native libraries are not included by default. add this if you want them (as of 0.7)
            // native libraries greatly improve performance, but increase jar sizes.
            "org.scalanlp" %% "breeze-natives" % "0.10"
)

resolvers ++= Seq(
            // other resolvers here
            // if you want to use snapshot builds (currently 0.11-SNAPSHOT), use this.
            "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
            "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)
